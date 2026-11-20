"""Local SGD, ER, GEM, A-GEM and a diagonal EWC baseline.

Client memory stores *training indices*, never copies of held-out test data.
GEM uses its actual dual QP and verifies primal constraints. Inaccurate
projections cause the job to fail instead of silently defaulting to SGD.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass,field
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,Dataset
from .cl_estimators import estimate_diagonal_importance
from .class_indexed import canonical_active_loss,freeze_other_class_rows
from scipy.optimize import minimize
from .native_methods import NATIVE_METHODS,native_loss,historical_teacher

@dataclass
class LocalState:
    client_id:int
    model_version:int=-1
    shared_weights:dict|None=None
    current_indices:list[int]=field(default_factory=list)
    current_stage:int=-1
    observed_classes:set[int]=field(default_factory=set)
    seen_indices:set[int]=field(default_factory=set)
    first_seen_stage:dict[int,int]=field(default_factory=dict)
    memory:dict[int,list[int]]=field(default_factory=dict)
    last_round:int=-1
    last_selected:int=-1
    recurrence_rounds:dict[int,int]=field(default_factory=dict)
    ewc_anchor:dict|None=None
    ewc_fisher:dict|None=None
    train_steps:int=0
    selection_count:int=0
    trained_classes:set[int]=field(default_factory=set)
    memory_logits:dict[int,torch.Tensor]=field(default_factory=dict)
    mas_omega:dict|None=None
    mas_anchor:dict|None=None
    lwf_teacher:dict|None=None
    lwf_stage:int=-1
    ewc_stage:int=-1
    sample_first_stage:dict[int,int]=field(default_factory=dict)
    optimizer_state:dict|None=None
    last_constraint_violations:int=0
    si_omega:dict|None=None
    si_anchor:dict|None=None
    si_path:dict|None=None
    si_displacement:dict|None=None
    si_stage:int=-1
    mas_stage:int=-1
    # The client always continues learning on its private model. A server
    # snapshot is an anchor for computing its entire, potentially stale delta.
    sync_anchor:dict|None=None
    pending_upload_examples:int=0
    pending_upload_class_counts:dict[int,int]=field(default_factory=dict)
    pending_upload_classes:set[int]=field(default_factory=set)
    pending_upload_steps:int=0
    pending_class_update_steps:dict[int,int]=field(default_factory=dict)
    activated_labels:set[int]=field(default_factory=set)
    local_experience:int=-1
    first_seen_local_experience:dict[int,int]=field(default_factory=dict)
    local_training_events:list[dict]=field(default_factory=list)
    private_memory:dict[int,list[int]]=field(default_factory=dict)
    private_reservoir_seen_counts:dict[int,int]=field(default_factory=dict)
    private_reservoir_seen_indices:set[int]=field(default_factory=set)
    received_memory:dict[int,list[int]]=field(default_factory=dict)
    received_example_payloads:dict[int,tuple]=field(default_factory=dict)
    received_gradient_supports:dict[int,tuple[int,...]]=field(default_factory=dict)
    pending_importance:dict[str,dict]=field(default_factory=dict)
    federated_cl_version:int=0
    lwf_teacher_from_server:bool=False
    # Class/experience provenance makes GEM's historical constraint groups
    # unambiguous even when tasks originate on different clients.
    sample_first_local_experience:dict[int,int]=field(default_factory=dict)
    received_sample_origins:dict[int,tuple[int,int]]=field(default_factory=dict)
    received_importance:dict[str,dict]=field(default_factory=dict)
    received_memory_logits:dict[int,torch.Tensor]=field(default_factory=dict)
    received_server_class_sample_exposure:dict[int,int]=field(default_factory=dict)
    received_lwf_teacher:dict|None=None
    received_lwf_teacher_version:int=-1
    last_learning_method:str=''
    native_teacher:dict|None=None
    native_teacher_classes:set[int]=field(default_factory=set)
    pending_feat_ecc:list[dict]=field(default_factory=list)
    received_feat_ecc:dict|None=None
    def receive(self,weights,version,*,clear_memory=False):
        if clear_memory:
            self.memory.clear()
            self.memory_logits.clear()
        self.shared_weights={k:v.detach().cpu().clone() for k,v in weights.items()}
        self.sync_anchor={k:v.detach().cpu().clone() for k,v in weights.items()}
        self.model_version=version
        # Acknowledged local contributions have now reached the server.
        self.pending_upload_examples=0
        self.pending_upload_class_counts={}
        self.pending_upload_classes=set()
        self.pending_upload_steps=0
        self.pending_class_update_steps={}
        self.pending_importance={}
        self.pending_feat_ecc=[]
    def arrival(self,indices,labels,stage):
        if stage > self.current_stage:
            self.current_indices=[]
            self.current_stage=stage
        self.current_indices.extend(indices)
        for idx in indices:self.sample_first_stage.setdefault(int(idx),int(stage))
        self.seen_indices.update(indices)
        self.observed_classes.update(labels)
        for cls in labels:
            self.first_seen_stage.setdefault(int(cls),int(stage))
    def update_memory(self,indices,targets,budget,rng,policy='random_per_class',*,
                      reservoir_fraction=0.0,class_totals=None):
        """Update the private buffer; never prune the synchronized server archive.

        With a positive fraction, Algorithm R samples independently per class
        using a fixed k=ceil(fraction * number of available TRAIN examples of
        that class). For the official CIFAR-100 pilot, 450 TRAIN/class => k=45.
        A fixed k makes the reservoir uniform over distinct examples seen so far;
        the local seen-ID ledger prevents repeated observations being double counted.
        """
        from .federated_cl import combined_memory
        if reservoir_fraction:
            if not 0 < reservoir_fraction <= 1:
                raise ValueError('Reservoir fraction must lie in (0,1]')
            if class_totals is None:
                raise ValueError('Classwise reservoir requires TRAIN class counts')
            from math import ceil
            for idx in indices:
                idx=int(idx)
                if idx in self.private_reservoir_seen_indices:
                    continue
                cls=int(targets[idx])
                size=int(class_totals.get(cls,0))
                if size<=0:
                    raise ValueError('No available TRAIN examples for class {}'.format(cls))
                self.private_reservoir_seen_indices.add(idx)
                count=self.private_reservoir_seen_counts.get(cls,0)+1
                self.private_reservoir_seen_counts[cls]=count
                k=ceil(reservoir_fraction*size)
                group=self.private_memory.setdefault(cls,[])
                if len(group)<k:
                    group.append(idx)
                else:
                    slot=int(rng.integers(0,count))
                    if slot<k:
                        group[slot]=idx
        else:
            by_class=defaultdict(list)
            for idx in indices:
                by_class[int(targets[idx])].append(int(idx))
            if budget>0:
                if policy=='random_per_class':
                    # Exact per-class Algorithm R across multiple experiences,
                    # rather than repeatedly re-sampling a previously reduced
                    # pool (which biases the reservoir toward recent arrivals).
                    for idx in indices:
                        idx=int(idx)
                        if idx in self.private_reservoir_seen_indices:continue
                        self.private_reservoir_seen_indices.add(idx)
                        cls=int(targets[idx])
                        seen=self.private_reservoir_seen_counts.get(cls,0)+1
                        self.private_reservoir_seen_counts[cls]=seen
                        group=self.private_memory.setdefault(cls,[])
                        if len(group)<budget:
                            group.append(idx)
                        else:
                            slot=int(rng.integers(0,seen))
                            if slot<budget:group[slot]=idx
                elif policy=='recent_per_class':
                    for cls,samples in by_class.items():
                        pool=list(dict.fromkeys(self.private_memory.get(cls,[])+samples))
                        self.private_memory[cls]=pool[-budget:]
                else:
                    raise ValueError(f'Unsupported memory sampling policy: {policy}')
        self.memory=combined_memory(self.private_memory,self.received_memory)
        retained={int(i) for group in self.memory.values() for i in group}
        for i in tuple(self.memory_logits):
            if i not in retained:
                del self.memory_logits[i]



def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def _flatten_grads(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).detach().clone()

def _set_grads(params, vector):
    pos=0
    for p in params:
        size=p.numel()
        p.grad=vector[pos:pos+size].reshape_as(p).clone()
        pos+=size
    assert pos==vector.numel()

def gem_project(gradient, references, tolerance=1e-5):
    """Project a parameter-space gradient onto all halfspaces g_j·z >= 0.

    Uses the mathematically equivalent convex nonnegative dual QP.
    Computation is done in CPU double precision for numerical stability.
    The calling local trainer performs normal model forward/backward on CUDA.
    """
    if not references:return gradient,0.,0
    G=torch.stack(references,dim=0).detach().to(device='cpu',dtype=torch.float64).numpy()
    g=gradient.detach().to(device='cpu',dtype=torch.float64).numpy()
    # A zero constraint is vacuous.
    mask=np.linalg.norm(G,axis=1)>1e-12
    G=G[mask]
    if not len(G):return gradient,0.,0
    violations=int(np.sum(G@g < -tolerance))
    if not violations:return gradient,0.,0
    Q=G@G.T
    b=G@g
    def obj(lam):return float(.5*lam@Q@lam + b@lam)
    def jac(lam):return Q@lam + b
    res=minimize(obj,np.zeros(len(G)),jac=jac,method='L-BFGS-B',bounds=[(0,None)]*len(G),
                 options={'maxiter':5000,'ftol':1e-13,'gtol':1e-11})
    z=g+G.T@res.x
    violation=float(max(0.,-np.min(G@z)))
    if violation>tolerance:
        # SLSQP on small dual can rescue some poorly conditioned cases.
        res=minimize(obj,np.maximum(res.x,0.),jac=jac,method='SLSQP',
                     bounds=[(0,None)]*len(G),options={'maxiter':2000,'ftol':1e-12})
        z=g+G.T@res.x
        violation=float(max(0.,-np.min(G@z)))
    if not np.all(np.isfinite(z)) or violation>tolerance:
        raise RuntimeError(f'GEM QP infeasible numerically: residual={violation:.4g}')
    return torch.as_tensor(z,device=gradient.device,dtype=gradient.dtype),float(np.linalg.norm(z-g)),violations


def agem_project(g, reference):
    denom=torch.dot(reference,reference)
    if denom.item() < 1e-20:return g,0.,0
    dot=torch.dot(g,reference)
    if dot.item() >= 0:return g,0.,0
    z=g-(dot/denom)*reference
    return z,float(torch.linalg.vector_norm(z-g).item()),1


class _IndexedDataset(Dataset):
    """Expose original TRAIN sample IDs for online DER logit capture."""
    def __init__(self,dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        x,y=self.dataset[index]
        return x,y,int(index)


class _CIFARCropFlipDataset(Dataset):
    """Apply standard CIFAR random crop and horizontal flip to a train view."""
    def __init__(self,dataset,rng,padding=4):
        self.dataset=dataset
        self.rng=rng
        self.padding=int(padding)
        from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
        self.mean=torch.tensor(NORMALIZATION_MEAN,dtype=torch.float32).view(3,1,1)
        self.std=torch.tensor(NORMALIZATION_STD,dtype=torch.float32).view(3,1,1)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        x,y=self.dataset[index]
        if not torch.is_tensor(x) or tuple(x.shape)!=(3,32,32):
            raise ValueError('CIFAR crop/flip expects normalized (3,32,32) tensors')
        x=(x*self.std+self.mean).clamp(0,1)
        p=self.padding
        if p:x=F.pad(x,(p,p,p,p),mode='constant',value=0.)
        top=int(self.rng.integers(0,2*p+1)) if p else 0
        left=int(self.rng.integers(0,2*p+1)) if p else 0
        x=x[:,top:top+32,left:left+32]
        if self.rng.random()<.5:x=x.flip(-1)
        return (x-self.mean)/self.std,y


class _FixedBatchSampler:
    """No short optimizer batches: replay final samples from the same TRAIN pool."""
    def __init__(self,size,batch_size,seed,shuffle):
        if size<1 or batch_size<1:raise ValueError('Empty training batch source')
        self.size,self.batch_size,self.seed,self.shuffle=size,batch_size,seed,shuffle
    def __len__(self):
        from math import ceil
        return ceil(self.size/self.batch_size)
    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed)
        order=(torch.randperm(self.size,generator=gen) if self.shuffle else torch.arange(self.size))
        for start in range(0,self.size,self.batch_size):
            selection=order[start:start+self.batch_size].tolist()
            if len(selection)<self.batch_size:
                extra=torch.randint(self.size,(self.batch_size-len(selection),),generator=gen)
                selection.extend(order[extra].tolist())
            assert len(selection)==self.batch_size
            yield selection


def _example_loader(dataset,indices,batch_size,seed,shuffle=True,*,include_indices=False,full_batch=False):
    source=_IndexedDataset(dataset) if include_indices else dataset
    subset=Subset(source,list(indices))
    if full_batch:
        sampler=_FixedBatchSampler(len(subset),batch_size,seed,shuffle)
        return DataLoader(subset,batch_sampler=sampler,num_workers=0,
                          pin_memory=torch.cuda.is_available())
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(subset,batch_size=batch_size,shuffle=shuffle,
                      num_workers=0,generator=generator,pin_memory=torch.cuda.is_available())


def _reference_gradient(model,params,dataset,indices,batch_size,device,seed,active_pair_classes=None,reference_classes=None):
    model.zero_grad(set_to_none=True)
    # Reference gradients are read-only with respect to BatchNorm running
    # statistics. Otherwise A-GEM/GEM accidentally trains CNN buffers on
    # reference examples, changing the federated model even if no replay loss
    # was requested. Retain training-mode gradients, restore only buffers.
    bn_snapshots=[]
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm) and layer.track_running_stats:
            bn_snapshots.append((layer,layer.running_mean.detach().clone(),
                layer.running_var.detach().clone(),layer.num_batches_tracked.detach().clone()))
    try:
        # One true physical reference batch; scarce history is drawn with replacement.
        count=len(indices)
        if not count:raise ValueError('No historical constraint samples')
        gen=torch.Generator().manual_seed(seed)
        draw=(torch.randperm(count,generator=gen)[:batch_size] if count>=batch_size
              else torch.randint(count,(batch_size,),generator=gen))
        selected=[indices[int(i)] for i in draw]
        total=len(selected)
        for x,y in _example_loader(dataset,selected,batch_size,seed,shuffle=False):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            reference_logits=model(x)
            if reference_classes is not None:
                # GEM explicit original-experience constraints retain their
                # own historical label universe (stage grouping).
                from .class_indexed import canonical_active_loss
                ref_loss=canonical_active_loss(reference_logits,y,reference_classes)
            elif active_pair_classes is not None:
                # All/class grouping and A-GEM must use the SAME known-class
                # competitor set as the current class-IL objective, rather
                # than penalizing outputs for unknown future classes.
                from .class_indexed import canonical_active_loss
                ref_loss=canonical_active_loss(reference_logits,y,active_pair_classes)
            else:
                ref_loss=F.cross_entropy(reference_logits,y)
            (ref_loss*(len(y)/total)).backward()
        if active_pair_classes is not None:
            from .class_indexed import freeze_other_class_rows
            freeze_other_class_rows(model,active_pair_classes)
        result=_flatten_grads(params)
    finally:
        with torch.no_grad():
            for layer,mean,variance,count in bn_snapshots:
                layer.running_mean.copy_(mean)
                layer.running_var.copy_(variance)
                layer.num_batches_tracked.copy_(count)
    return result


def local_train(model,client,bank,cfg,device,round_id,task_classes=None,*,experience_end=False,active_pair_classes=None):
    """Continue training on the client's own local model, without server access.

    The simulator calls this for every client with local data every round,
    whether selected for federation or not. Shared weights are the private
    local model between rounds; server synchronization is handled separately.
    """
    if cfg.model=='fedclass_resnet8' and cfg.method in {'replay','sgd'}:
        from .upstream_fedclass_ce import train_original_ce
        return train_original_ce(model,client,bank,cfg,device,round_id,
                                 task_classes,experience_end=experience_end,
                                 active_pair_classes=active_pair_classes)
    if client.shared_weights is None:
        raise RuntimeError('Client must synchronize before local training')
    client.last_learning_method=cfg.method
    model.load_state_dict(client.shared_weights)
    model.to(device)
    model.train()
    if active_pair_classes is not None:
        active_pair_classes=tuple(sorted(set(int(c) for c in active_pair_classes)))
        if len(active_pair_classes)<2:
            raise ValueError('Local task requires at least two distinct labels')
    # Class-incremental local objective: new images must compete with every
    # class already known to this client, including freshly synchronized
    # server classes. Future classes alone remain inactive/frozen.
    seen_classes=tuple(sorted(set(client.observed_classes) |
                              set(active_pair_classes or ())))
    if active_pair_classes is not None and not set(active_pair_classes).issubset(seen_classes):
        raise RuntimeError('Active class pair was lost from client CL history')
    indices=list(dict.fromkeys(client.current_indices))
    if not indices:
        return None
    rng=np.random.default_rng(cfg.seed+round_id*137+client.client_id*100003)
    training_data=bank.train
    if cfg.scenario in {'domain','gradual_drift','shifted_recurrence'} or cfg.label_noise_rate or cfg.spurious_strength:
        from .domains import ClientAugmentedDataset
        training_data=ClientAugmentedDataset(bank.train,client.client_id,
                    client.sample_first_stage,cfg.scenario,
                    spurious_strength=cfg.spurious_strength,
                    label_noise_rate=cfg.label_noise_rate,seed=cfg.seed,
                    num_classes=bank.num_classes)
    if cfg.share_continual_state and client.received_example_payloads:
        # Historical samples from other clients are actually read from the
        # synchronized image archive, not the simulator's privileged data bank.
        from .federated_cl import ClientArchiveDataset
        training_data=ClientArchiveDataset(training_data,client.received_example_payloads)
    if cfg.training_augmentation=='cifar_crop_flip':
        augmentation_rng=np.random.default_rng(
            cfg.seed+4_170_001+round_id*100_003+client.client_id*10_007+
            (client.local_experience+1)*101)
        training_data=_CIFARCropFlipDataset(training_data,augmentation_rng)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    if cfg.optimizer_state_policy=='persist' and client.optimizer_state is not None:
        optimizer.load_state_dict(client.optimizer_state)
    params=_trainable_params(model)
    native_teacher=(historical_teacher(model,client.native_teacher,device)
                    if cfg.method in {'native_glfc','native_lga'} and client.native_teacher is not None
                    else None)
    native_old_classes=set(client.native_teacher_classes)&set(seen_classes)
    if active_pair_classes:
        native_old_classes.difference_update(active_pair_classes)
    if cfg.method=='si':
        if client.si_omega is None:
            client.si_omega={}
        for n,p in model.named_parameters():
            client.si_omega.setdefault(n,torch.zeros_like(p.detach().cpu()))
        if client.si_anchor is not None:
            for n,p in model.named_parameters():
                client.si_anchor.setdefault(n,p.detach().cpu().clone())
        if client.si_path is None:
            client.si_path={n:torch.zeros_like(p.detach().cpu())
                            for n,p in model.named_parameters()}
        if client.si_displacement is None:
            client.si_displacement={n:torch.zeros_like(p.detach().cpu())
                                    for n,p in model.named_parameters()}
        if (client.si_stage>=0 and client.local_experience>client.si_stage
                and any(torch.count_nonzero(x).item() for x in client.si_displacement.values())):
            fresh_importance={}
            # SI consolidation uses ONLY locally applied optimization steps;
            # the global broadcast is deliberately excluded from the path.
            for n,p in model.named_parameters():
                delta=client.si_displacement[n]
                importance=(client.si_path[n]/(delta.square()+cfg.si_xi)).clamp(min=0)
                fresh_importance[n]=importance
                client.si_path[n].zero_()
                client.si_displacement[n].zero_()
            current_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            from .federated_cl import add_importance,weighted_importance
            add_importance(client,'si',fresh_importance,current_anchor)
            client.si_omega,client.si_anchor=weighted_importance(
                client.si_omega,client.si_anchor,fresh_importance,current_anchor)
        elif client.si_anchor is None:
            client.si_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        client.si_stage=client.local_experience
    # Persist an earlier global teacher across all rounds of the new local stage.
    if cfg.method=='lwf' and client.local_experience>client.lwf_stage:
        if client.lwf_teacher_from_server:
            # Use the global historical teacher once after re-synchronization.
            client.lwf_teacher_from_server=False
        else:
            client.lwf_teacher={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} \
                if client.lwf_stage>=0 else None
        client.lwf_stage=client.local_experience
    teacher=None
    if cfg.method=='lwf' and client.lwf_teacher is not None:
        import copy
        teacher=copy.deepcopy(model)
        teacher.load_state_dict(client.lwf_teacher)
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    # FedProx regularizes toward the *last downloaded global model*.
    # A disconnected client cannot use a later global snapshot it has never
    # received, and its unsent trajectory must not be silently overwritten.
    anchor={name:(client.sync_anchor[name].to(device).detach().clone()
                  if cfg.fedprox_mu>0 and client.sync_anchor is not None
                  else p.detach().clone())
            for name,p in model.named_parameters()}
    # FedProx is a local objective, not a different aggregation formula.

    # Rehearsal and constraint memory are built only from previous locally trained examples.
    eligible_history={cls:inds for cls,inds in client.memory.items()
                      if (cls in client.trained_classes if cfg.boundary_mode=='hidden'
                          else client.first_seen_local_experience.get(cls,client.local_experience)<client.local_experience)}
    oldmemory=[i for group in eligible_history.values() for i in group]
    if cfg.method in {'der','derpp'}:
        # A heterogeneous remote replay archive can contain images from
        # clients running other methods that never stored trajectory logits.
        # Do not hallucinate model-logit targets or crash on missing targets.
        oldmemory=[idx for idx in oldmemory if idx in client.memory_logits]
    def local_old_loss():
        if not oldmemory:return None
        was_train=model.training
        model.eval()
        with torch.no_grad():
            subset=oldmemory[:min(cfg.batch_size,len(oldmemory))]
            valid=len(subset)
            batch=torch.stack([training_data[i][0] for i in subset]).to(device)
            if valid<cfg.batch_size:
                duplicates=torch.arange(cfg.batch_size-valid,device=device)%valid
                batch=torch.cat((batch,batch[duplicates]),dim=0)
            labels=torch.tensor([bank.train.targets[i] for i in subset],device=device)
            # Do not let added computational padding bias loss diagnostics.
            logits=model(batch)[:valid]
            value=float(F.cross_entropy(logits,labels).item())
        model.train(was_train)
        return value
    old_loss_before=local_old_loss() if cfg.prepost_diagnostics else None
    step_count=0; loss_total=0.; last_projection=0.; num_violations=0;qp_calls=0
    def draw_history(n,tag):
        """CPU retrieval for assembly into a single physical model input of 64."""
        chosen=rng.choice(oldmemory,size=n,replace=len(oldmemory)<n).tolist()
        x,y=next(iter(_example_loader(training_data,chosen,n,
                     cfg.seed+round_id+client.client_id+tag,shuffle=False)))
        return x,y,chosen

    def full_training_batches(loaded):
        x,y=loaded[:2]
        original_ids=loaded[2] if len(loaded)>2 else None
        if cfg.method not in ({'replay'}|NATIVE_METHODS) or not oldmemory or cfg.replay_ratio<=0:
            yield x,y,original_ids
            return
        # One supervised CE over a MIXED current/history physical batch of 64.
        # This eliminates doubled separately averaged Replay gradients.
        fresh=max(1,min(cfg.batch_size-1,int(round(cfg.batch_size/(1+cfg.replay_ratio)))))
        for offset in range(0,len(y),fresh):
            nx,ny=x[offset:offset+fresh],y[offset:offset+fresh]
            if len(ny)<fresh:
                fill=torch.as_tensor(rng.integers(0,len(y),size=fresh-len(ny)),dtype=torch.long)
                nx=torch.cat((nx,x[fill]),dim=0)
                ny=torch.cat((ny,y[fill]),dim=0)
            hx,hy,_=draw_history(cfg.batch_size-fresh,offset)
            yield torch.cat((nx,hx),dim=0),torch.cat((ny,hy),dim=0),None

    for epoch in range(cfg.epochs):
        online_der=cfg.method in {'der','derpp'}
        for loaded in _example_loader(training_data,indices,cfg.batch_size,
                                   cfg.seed+round_id*100_003+client.client_id*127+epoch,
                                   include_indices=online_der,full_batch=True):
            for x,y,original_ids in full_training_batches(loaded):
                optimizer.zero_grad(set_to_none=True)
                logical_size=len(y)
                macro_loss=0.
                si_task_grads=None
                captured_logits={}
                if logical_size!=cfg.batch_size:
                    raise RuntimeError('Training forward does not have exactly 64 images')
                x=x.to(device,non_blocking=True)
                y=y.to(device,non_blocking=True)
                if cfg.method in NATIVE_METHODS:
                    loss,logits=native_loss(cfg.method,model,x,y,seen_classes,
                                            teacher=native_teacher,old_classes=native_old_classes)
                else:
                    logits=model(x)
                if original_ids is not None:
                    for idx,z in zip(original_ids.tolist(),logits.detach().cpu()):
                        captured_logits[int(idx)]=z.clone()
                if cfg.method not in NATIVE_METHODS:
                    if active_pair_classes is not None:
                        # ER-ACE is intentionally asymmetric in its source paper:
                        # mask historical logits for *new* examples only, unlike
                        # standard ER, EWC, SI, MAS, GEM, LwF and DER.
                        current_competition=(active_pair_classes if cfg.method=='er_ace'
                                             else seen_classes)
                        loss=canonical_active_loss(logits,y,current_competition)
                    elif cfg.learning_setting=='task_il':
                        if task_classes is None or not set(y.tolist()).issubset(set(task_classes)):
                            raise RuntimeError('Task-IL labels do not match the known local task')
                        task_mask=torch.ones(logits.shape[1],dtype=torch.bool,device=device)
                        task_mask[torch.as_tensor(task_classes,dtype=torch.long,device=device)]=False
                        logits=logits.masked_fill(task_mask.unsqueeze(0),-1e9)
                    if active_pair_classes is None and cfg.method=='er_ace':
                        # ER-ACE stream loss: mask historical classes absent from the
                        # *current* physical batch, but retain future/unseen output columns.
                        mask=torch.zeros(logits.shape[1],dtype=torch.bool,device=device)
                        current_labels=set(int(v) for v in y.unique().tolist())
                        for past_cls in client.observed_classes:
                            if past_cls not in current_labels and past_cls < logits.shape[1]:
                                mask[past_cls]=True
                        stream_logits=logits.masked_fill(mask.unsqueeze(0),-1e9)
                        loss=F.cross_entropy(stream_logits,y)
                    elif active_pair_classes is None:
                        loss=F.cross_entropy(logits,y)
                if cfg.method=='si':
                    # SI path integral uses *unregularized current task loss*
                    # (not SI/EWC/FedProx penalties or projected gradients).
                    task_grad=torch.autograd.grad(loss,params,retain_graph=True,
                                                  allow_unused=True)
                    if si_task_grads is None:
                        si_task_grads=[torch.zeros_like(p.detach(),device='cpu')
                                       for p in params]
                    for j,(parameter,grad) in enumerate(zip(params,task_grad)):
                        if grad is not None:
                            si_task_grads[j]+=grad.detach().cpu()*(len(y)/logical_size)
                if cfg.method in {'er_ace','der','derpp'} and oldmemory:
                    ref_count=cfg.batch_size
                    rx,ry,chosen=draw_history(ref_count,1000)
                    rx,ry=rx.to(device,non_blocking=True),ry.to(device,non_blocking=True)
                    replay_logits=model(rx)
                    if cfg.method=='er_ace':
                        loss=loss+cfg.replay_ratio*cfg.der_beta*(
                            canonical_active_loss(replay_logits,ry,seen_classes)
                            if active_pair_classes is not None else F.cross_entropy(replay_logits,ry))
                    else:
                        missing=[int(i) for i in chosen if int(i) not in client.memory_logits]
                        if missing:
                            raise RuntimeError(f'DER memory logits missing for {missing[:3]}')
                        old_logits=torch.stack([client.memory_logits[int(i)] for i in chosen]).to(device)
                        if old_logits.shape!=replay_logits.shape:
                            raise RuntimeError('DER logits incompatible with classifier schema')
                        loss=loss+cfg.replay_ratio*cfg.der_alpha*F.mse_loss(replay_logits,old_logits)
                        if cfg.method=='derpp':
                            # DER++ draws TWO independent full-size historical batches:
                            # one for old trajectory-logit MSE, one independently
                            # for supervised old-label CE. Sharing a single draw
                            # couples their gradient estimators unnecessarily.
                            bx,by,cls_idx=draw_history(ref_count,1913)
                            supervised_logits=model(bx.to(device))
                            loss=loss+cfg.replay_ratio*cfg.der_beta*(
                                canonical_active_loss(supervised_logits,by.to(device),seen_classes)
                                if active_pair_classes is not None else F.cross_entropy(supervised_logits,by.to(device)))
                if cfg.method=='lwf' and teacher is not None:
                    with torch.no_grad():t_logits=teacher(x)
                    old_labels=sorted(c for c,s in client.first_seen_local_experience.items()
                                      if s < client.local_experience)
                    if old_labels:
                        ids=torch.tensor(old_labels,dtype=torch.long,device=device)
                        temperature=2.0
                        student=F.log_softmax(logits.index_select(1,ids)/temperature,dim=1)
                        past=F.softmax(t_logits.index_select(1,ids)/temperature,dim=1)
                        loss=loss+cfg.der_alpha*temperature**2*F.kl_div(student,past,reduction='batchmean')
                if cfg.method=='si' and client.si_anchor is not None:
                    penalty=torch.zeros((),device=device)
                    for name,p in model.named_parameters():
                        penalty=penalty+(client.si_omega[name].to(device)*
                                         (p-client.si_anchor[name].to(device)).square()).sum()
                    loss=loss+.5*cfg.regularization_strength*penalty
                if cfg.method=='mas' and client.mas_anchor is not None:
                    penalty=torch.zeros((),device=device)
                    for name,p in model.named_parameters():
                        if name in client.mas_anchor:
                            penalty=penalty+(client.mas_omega[name].to(device)*
                                             (p-client.mas_anchor[name].to(device)).square()).sum()
                    loss=loss+.5*cfg.regularization_strength*penalty
                if cfg.fedprox_mu>0:
                    proximal=sum((p-anchor[name]).square().sum() for name,p in model.named_parameters())
                    loss=loss+.5*cfg.fedprox_mu*proximal
                if cfg.method=='ewc' and client.ewc_anchor is not None:
                    pen=torch.zeros((),device=device)
                    for name,p in model.named_parameters():
                        if name in client.ewc_anchor:
                            ewc_reference=client.ewc_anchor[name].to(device)
                            fisher=client.ewc_fisher[name].to(device)
                            pen=pen+(fisher*(p-ewc_reference).square()).sum()
                    loss=loss+.5*cfg.regularization_strength*pen
                macro_loss+=float(loss.detach().item())*(len(y)/logical_size)
                (loss*(len(y)/logical_size)).backward()
                if cfg.method in {'gem','agem'} and oldmemory:
                    if active_pair_classes is not None:
                        freeze_other_class_rows(model,seen_classes)
                    g=_flatten_grads(params)
                    if cfg.method=='agem':
                        refidx=rng.choice(oldmemory,size=min(cfg.batch_size,len(oldmemory)),replace=False).tolist()
                        ref=_reference_gradient(model,params,training_data,refidx,cfg.batch_size,device,
                                                cfg.seed+round_id+client.client_id,seen_classes)
                        z,last_projection,v=agem_project(g,ref)
                    else:
                        refs=[]
                        # Explicit constraint grouping; hidden experience labels
                        # are not passed to task-free learning algorithms.
                        if cfg.gem_grouping=='class':
                            groups=eligible_history
                        elif cfg.gem_grouping=='stage':
                            groups=defaultdict(list)
                            for cls,mem in eligible_history.items():
                                for idx in mem:
                                    # The remote exemplar carries the *source* client's
                                    # experience ID; default lookups must not raise for
                                    # remotely trained classes absent locally.
                                    source=client.received_sample_origins.get(int(idx))
                                    if source is None:
                                        source=(client.client_id,client.sample_first_local_experience.get(
                                            int(idx),client.sample_first_stage.get(int(idx),
                                                client.first_seen_stage.get(cls,-1))))
                                    groups[source].append(idx)
                        else:
                            groups={0:oldmemory}
                        for group in sorted(groups):
                            mem=groups[group]
                            if mem:
                                historical_classes=(tuple(sorted(set(int(bank.train.targets[int(i)]) for i in mem)))
                                    if cfg.gem_grouping=='stage' else None)
                                if historical_classes is not None and len(historical_classes)<2:
                                    historical_classes=None  # sparse class-imbalanced memory
                                refs.append(_reference_gradient(model,params,training_data,mem,
                                                                cfg.batch_size,device,cfg.seed+(group if isinstance(group,int) else (group[0]*100003+group[1])),seen_classes,
                                                                historical_classes))
                        z,last_projection,v=gem_project(g,refs)
                        qp_calls+=int(v>0)
                    num_violations+=v
                    _set_grads(params,z)
                if active_pair_classes is not None:
                    freeze_other_class_rows(model,seen_classes)
                frozen=None
                if active_pair_classes is not None:
                    from .class_indexed import snapshot_frozen_rows,restore_frozen_rows
                    frozen=snapshot_frozen_rows(model,seen_classes)
                if cfg.method=='si':
                    before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                    # SI uses the unregularized data-loss path integral.
                    applied_grad={n:si_task_grads[j].clone()
                                  for j,(n,p) in enumerate(model.named_parameters())}
                optimizer.step()
                if frozen is not None:
                    restore_frozen_rows(model,optimizer,frozen)
                if cfg.method=='si':
                    for n,p in model.named_parameters():
                        displacement=p.detach().cpu()-before[n]
                        client.si_path[n]+=-applied_grad[n]*displacement
                        client.si_displacement[n]+=displacement
                if online_der:
                    # Online DER inserts the model's PRE-UPDATE output at the
                    # moment an input enters the episodic reservoir. The original
                    # logit target must not be overwritten by a later checkpoint.
                    batch_new_ids=list(map(int,original_ids.tolist()))
                    client.update_memory(batch_new_ids,bank.train.targets,
                        cfg.memory_per_class,rng,cfg.memory_policy)
                    retained={int(i) for group in client.private_memory.values() for i in group}
                    for idx in retained:
                        if idx in captured_logits and idx not in client.memory_logits:
                            client.memory_logits[idx]=captured_logits[idx]
                step_count+=1
                loss_total+=macro_loss
    # Finalize the just-completed SI experience *before* communication. In v7,
    # consolidation happened only at the next arrival; the most recent task's
    # importance consequently never reached the server in that sync.
    if cfg.method=='si' and experience_end and client.si_path is not None:
        fresh={}
        for n,p in model.named_parameters():
            displacement=client.si_displacement[n]
            fresh[n]=(client.si_path[n]/(displacement.square()+cfg.si_xi)).clamp(min=0)
            client.si_path[n].zero_()
            client.si_displacement[n].zero_()
        fresh_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        from .federated_cl import add_importance,weighted_importance
        add_importance(client,'si',fresh,fresh_anchor)
        client.si_omega,client.si_anchor=weighted_importance(
            client.si_omega,client.si_anchor,fresh,fresh_anchor)
        client.si_stage=client.local_experience
    # EWC and MAS must average individual-sample importance contributions.
    # Squaring or abs() AFTER averaging a minibatch gradient is biased and
    # allows gradients from different examples to cancel.
    if cfg.method in {'ewc','mas'} and experience_end:
        if cfg.method=='ewc' and client.local_experience<=client.ewc_stage:
            raise RuntimeError('Repeated EWC importance consolidation for one experience')
        if cfg.method=='mas' and client.local_experience<=client.mas_stage:
            raise RuntimeError('Repeated MAS importance consolidation for one experience')
        fresh=estimate_diagonal_importance(model,training_data,indices,device,
            kind=cfg.method,max_samples=cfg.importance_samples_per_experience,
            seed=cfg.seed+round_id*100003+client.client_id*127+client.local_experience,
            active_classes=(seen_classes if active_pair_classes is not None else None))
        fresh_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        from .federated_cl import add_importance,weighted_importance
        add_importance(client,cfg.method,fresh,fresh_anchor)
        if cfg.method=='ewc':
            client.ewc_fisher,client.ewc_anchor=weighted_importance(
                client.ewc_fisher,client.ewc_anchor,fresh,fresh_anchor)
            client.ewc_stage=client.local_experience
        else:
            client.mas_omega,client.mas_anchor=weighted_importance(
                client.mas_omega,client.mas_anchor,fresh,fresh_anchor)
            client.mas_stage=client.local_experience
    # Never allocate replay/GEM memory for non-memory algorithms.
    if cfg.method in ({'gem','agem','replay','er_ace'}|NATIVE_METHODS):
        client.update_memory(indices,bank.train.targets,cfg.memory_per_class,rng,cfg.memory_policy,
                             reservoir_fraction=(cfg.replay_reservoir_fraction if cfg.method in ({'replay'}|NATIVE_METHODS) else 0.),
                             class_totals=(Counter(map(int,bank.train.targets)) if cfg.method in ({'replay'}|NATIVE_METHODS) and cfg.replay_reservoir_fraction else None))
    if cfg.method=='native_feat' and experience_end and active_pair_classes:
        # Original FEAT projection/EMA energies are computed from this
        # client's legitimately available examples, even while offline;
        # they cannot be published until the client reconnects.
        from .upstream_feat_geometry import compute_client_feat_ecc
        entry=compute_client_feat_ecc(model,client,bank,device,active_pair_classes,
                                      batch_size=cfg.batch_size)
        if entry is not None:
            client.pending_feat_ecc.append(entry)
    if cfg.method in NATIVE_METHODS and experience_end:
        # The next local experience (also when offline) uses this actual local
        # checkpoint as its distillation teacher, until a legitimate download.
        client.native_teacher={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        client.native_teacher_classes=set(seen_classes)
    old_loss_after=local_old_loss() if cfg.prepost_diagnostics else None
    client.last_constraint_violations=num_violations
    if cfg.optimizer_state_policy=='persist':
        client.optimizer_state=optimizer.state_dict()
        for state in client.optimizer_state['state'].values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):
                    state[key]=value.detach().cpu().clone()
    else:
        client.optimizer_state=None
    client.train_steps+=step_count
    client.trained_classes.update(int(bank.train.targets[i]) for i in indices)
    class_counts=dict(Counter(int(bank.train.targets[i]) for i in indices))
    if active_pair_classes is not None:
        # All seen classifier rows are trainable under v14 class-IL CE.
        # Custom row-wise server fusion must therefore account for every
        # row the client may have updated, not only today's two class IDs.
        # This ledger is UPDATE exposure, not fresh training-example count.
        client.activated_labels.update(seen_classes)
        class_update_steps={int(c):step_count for c in seen_classes}
    else:
        client.activated_labels.update(class_counts)
        class_update_steps={int(c):step_count for c in class_counts}

    return {'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'examples':len(indices),'class_counts':class_counts,'class_update_steps':class_update_steps,
            'active_pair_classes':list(active_pair_classes) if active_pair_classes else None,'steps':step_count,'loss':loss_total/max(1,step_count),
            'constraint_violations':num_violations,'qp_calls':qp_calls,
            'projection_norm_last':last_projection,'memory_samples':sum(map(len,client.memory.values())),
            'method_used':cfg.method,'fedprox_mu':cfg.fedprox_mu,
            'local_old_loss_before':old_loss_before,
            'local_old_loss_after':old_loss_after,
            'optimizer_state_policy':cfg.optimizer_state_policy,
            'sgd_momentum':cfg.momentum,'sgd_weight_decay':cfg.weight_decay,
            'classification_competitor_classes':list(seen_classes)}
