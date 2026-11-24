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
from .class_indexed import canonical_active_loss,freeze_other_class_rows,task_head_loss
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
    pending_task_head_examples:dict[str,float]=field(default_factory=dict)
    activated_labels:set[int]=field(default_factory=set)
    local_experience:int=-1
    first_seen_local_experience:dict[int,int]=field(default_factory=dict)
    local_training_events:list[dict]=field(default_factory=list)
    joint_local_source_heads:dict[int,set[str]]=field(default_factory=dict)
    joint_received_source_heads:dict[int,set[str]]=field(default_factory=dict)
    joint_pending_occurrences:list[str]=field(default_factory=list)
    private_memory:dict[int,list[int]]=field(default_factory=dict)
    private_reservoir_seen_counts:dict[int,int]=field(default_factory=dict)
    private_reservoir_seen_indices:set[int]=field(default_factory=set)
    received_memory:dict[int,list[int]]=field(default_factory=dict)
    received_example_payloads:dict[int,tuple]=field(default_factory=dict)
    received_gradient_supports:dict[int,tuple[int,...]]=field(default_factory=dict)
    pending_importance:dict[str,dict]=field(default_factory=dict)
    federated_cl_version:int=0
    received_replay_version:int=-1
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
    native_memory_state:dict=field(default_factory=dict)
    received_native_artifacts:dict[int,dict]=field(default_factory=dict)
    received_native_proxy:dict|None=None
    received_lga_radius:float|None=None
    pending_native_prototypes:list[dict]=field(default_factory=list)
    received_native_prototypes:dict=field(default_factory=dict)
    received_native_proxy:dict|None=None
    pending_feat_ecc:list[dict]=field(default_factory=list)
    feat_selection_tickets:dict[int,list[int]]=field(default_factory=dict)
    feat_selection_receipts:list[dict]=field(default_factory=list)
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
        self.pending_task_head_examples={}
        self.joint_pending_occurrences=[]
        self.pending_importance={}
        self.pending_feat_ecc=[]
        self.pending_native_prototypes=[]
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
    def __init__(self,size,batch_size,seed,shuffle,pad_pool_size=0):
        if size<1 or batch_size<1:raise ValueError('Empty training batch source')
        self.size,self.batch_size,self.seed,self.shuffle=size,batch_size,seed,shuffle
        self.pad_pool_size=int(pad_pool_size)
    def __len__(self):
        from math import ceil
        return ceil(self.size/self.batch_size)
    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed)
        order=(torch.randperm(self.size,generator=gen) if self.shuffle else torch.arange(self.size))
        for start in range(0,self.size,self.batch_size):
            selection=order[start:start+self.batch_size].tolist()
            if len(selection)<self.batch_size:
                missing=self.batch_size-len(selection)
                if self.pad_pool_size:
                    extra=torch.randint(self.pad_pool_size,(missing,),generator=gen)+self.size
                    selection.extend(extra.tolist())
                else:
                    extra=torch.randint(self.size,(missing,),generator=gen)
                    selection.extend(order[extra].tolist())
            assert len(selection)==self.batch_size
            yield selection


def _example_loader(dataset,indices,batch_size,seed,shuffle=True,*,include_indices=False,full_batch=False,padding_indices=None):
    source=_IndexedDataset(dataset) if include_indices else dataset
    primary=list(indices)
    padding=list(padding_indices or ())
    subset=Subset(source,primary+padding)
    if full_batch:
        sampler=_FixedBatchSampler(len(primary),batch_size,seed,shuffle,
                                   pad_pool_size=len(padding))
        return DataLoader(subset,batch_sampler=sampler,num_workers=0,
                          pin_memory=torch.cuda.is_available())
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(subset,batch_size=batch_size,shuffle=shuffle,
                      num_workers=0,generator=generator,pin_memory=torch.cuda.is_available())


def _task_ids_for_indices(indices,task_ids_by_index,fallback=()):
    if task_ids_by_index is None:
        return [tuple(fallback) for _ in indices]
    result=[]
    for index in indices:
        task_ids=task_ids_by_index.get(int(index),())
        if not task_ids:
            raise ValueError(f'No task provenance for replay sample {int(index)}')
        result.append(tuple(task_ids))
    return result


def _reference_gradient(model,params,dataset,indices,batch_size,device,seed,active_pair_classes=None,reference_classes=None,task_ids=None,task_ids_by_index=None):
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
        # Compute the reference direction over the complete stored buffer.
        # Mini-batches bound activation memory; the weighted gradient sum is
        # the same mean as one logical full-buffer batch.
        selected=list(indices)
        count=len(selected)
        if not count:raise ValueError('No historical constraint samples')
        total=len(selected)
        for x,y,ids in _example_loader(dataset,selected,batch_size,seed,shuffle=False,include_indices=True):
            valid=len(y)
            if valid<batch_size:
                pad=torch.arange(batch_size-valid)%valid
                x=torch.cat((x,x[pad]),dim=0)
                y=torch.cat((y,y[pad]),dim=0)
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            y=y[:valid]
            if hasattr(getattr(model,'classifier',None),'task_classes'):
                ref_loss=model.loss_for_labels(
                    x[:valid],y,task_ids=task_ids,
                    task_ids_by_example=_task_ids_for_indices(
                        ids[:valid].tolist(),task_ids_by_index,task_ids))
            else:
                reference_logits=model(x)[:valid]
                if reference_classes is not None:
                    # GEM explicit original-experience constraints retain
                    # their historical label universe (stage grouping).
                    from .class_indexed import canonical_active_loss
                    ref_loss=canonical_active_loss(reference_logits,y,reference_classes)
                elif active_pair_classes is not None and len(set(active_pair_classes))>=2:
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


def local_train(model,client,bank,cfg,device,round_id,task_classes=None,*,task_id=None,experience_end=False,active_pair_classes=None,padding_indices=None,known_task_ids=None,task_ids_by_index=None,current_head_by_index=None):
    """Continue training on the client's own local model, without server access.

    The simulator calls this for every client with local data every round,
    whether selected for federation or not. Shared weights are the private
    local model between rounds; server synchronization is handled separately.
    """
    if cfg.method=='native_feat' and cfg.learning_setting!='task_il':
        from .upstream_feat_train import train_original_feat
        return train_original_feat(
            model,client,bank,cfg,device,round_id,task_classes,
            experience_end=experience_end,
            active_pair_classes=active_pair_classes)
    if cfg.method in {'native_glfc','native_lga'} and cfg.learning_setting!='task_il':
        from .upstream_glfc_lga_train import train_original_glfc_lga
        return train_original_glfc_lga(
            model,client,bank,cfg,device,round_id,task_classes,
            experience_end=experience_end,
            active_pair_classes=active_pair_classes)
    if cfg.model=='fedclass_resnet8' and cfg.method in {'replay','sgd'} and cfg.learning_setting!='task_il':
        from .upstream_fedclass_ce import train_original_ce
        return train_original_ce(model,client,bank,cfg,device,round_id,
                                 task_classes,experience_end=experience_end,
                                 active_pair_classes=active_pair_classes)
    if client.shared_weights is None:
        raise RuntimeError('Client must synchronize before local training')
    client.last_learning_method=cfg.method
    model.load_state_dict(client.shared_weights)
    model.to(device)
    if cfg.learning_setting=='task_il':
        if not task_classes or not task_id:
            raise ValueError('Task-IL local training requires the active task ID')
        model.set_active_task_id(task_id)
    model.train()
    if current_head_by_index is not None and (cfg.joint_control_kind!='simultaneous_access' or cfg.learning_setting!='task_il'):
        raise ValueError('Mixed current task routing is exclusive to the pooled Task-IL control')
    if active_pair_classes is not None:
        active_pair_classes=tuple(sorted(set(int(c) for c in active_pair_classes)))
        if len(active_pair_classes)<2:
            raise ValueError('Local task requires at least two distinct labels')
    # Task-IL current examples use the active task head; replay routing uses
    # only task IDs that have arrived by this synchronization.
    seen_classes=tuple(sorted(set(client.observed_classes) |
                              set(active_pair_classes or ())))
    from .models import task_head_key
    known_task_ids={task_head_key(value) for value in (known_task_ids or ())}
    for event in client.local_training_events:
        if event.get('task_id'):
            known_task_ids.add(task_head_key(event['task_id']))
        elif event.get('classes'):
            known_task_ids.add(task_head_key(event['classes']))
    if task_id:known_task_ids.add(task_head_key(task_id))
    if cfg.learning_setting=='task_il':
        known_task_keys={task_head_key(task_id) for task_id in known_task_ids}
    else:
        known_task_keys=set()
    if active_pair_classes is not None and not set(active_pair_classes).issubset(seen_classes):
        raise RuntimeError('Active class pair was lost from client CL history')
    indices=list(dict.fromkeys(client.current_indices))
    if not indices:
        return None
    if current_head_by_index is not None and set(current_head_by_index)!=set(indices):
        raise ValueError('Pooled current head mapping must cover the exact fresh sample set')
    # LocalState starts at -1; arrival increments to zero for experience 1.
    rng_round=(client.local_experience if cfg.joint_plan_path else round_id)
    if rng_round<0:raise RuntimeError('Local training requires a delivered experience')
    rng=np.random.default_rng(cfg.seed+rng_round*137+client.client_id*100003)
    replay_order_rng=np.random.default_rng(
        cfg.seed+7_314_159+rng_round*1_000_003+client.client_id*10_007+
        max(0,int(client.local_experience)))
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
            cfg.seed+4_170_001+rng_round*100_003+client.client_id*10_007+
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

    # Build the local private + synchronized archive used by replay learners.
    eligible_history={cls:inds for cls,inds in client.memory.items()
                      if (cls in client.trained_classes if cfg.boundary_mode=='hidden'
                          else client.first_seen_local_experience.get(cls,client.local_experience)<client.local_experience)}
    if cfg.method in ({'replay','er_ace','der','derpp','gem','agem'}|NATIVE_METHODS):
        # Replay draws from every example in the client's currently available
        # private and synchronized archive. A disconnected client naturally
        # sees only the archive snapshot it last received.
        eligible_history=dict(client.memory)
    oldmemory=[]
    for cls,group in eligible_history.items():
        # Read FEAT's selected image IDs from its ticket table; duplicate
        # tickets are deduplicated below so each stored image contributes once.
        tickets=(client.feat_selection_tickets.get(cls) if cfg.method=='native_feat'
                 else None)
        oldmemory.extend(tickets if tickets is not None else group)
    # Replay each retained image once even if a source selector recorded the
    # same image through repeated tickets or duplicate class-table entries.
    oldmemory=list(dict.fromkeys(map(int,oldmemory)))
    replay_by_class={int(cls):list(dict.fromkeys(map(int,group)))
                     for cls,group in eligible_history.items() if group}
    replay_sampled_class_counts=Counter()
    replay_task_head_exposure=Counter()
    def record_task_replay(sample_ids,scale=1.):
        if cfg.learning_setting!='task_il':return
        from .models import task_head_key
        for index in sample_ids:
            ids=_task_ids_for_indices([index],task_ids_by_index,known_task_ids)[0]
            keys=tuple(sorted({task_head_key(task_id) for task_id in ids}&known_task_keys))
            if not keys:
                raise ValueError(f'Replay sample {int(index)} has no task head')
            for key in keys:replay_task_head_exposure[key]+=float(scale)/len(keys)
    def draw_history(count):
        """Sample classes and images uniformly without avoidable repeats."""
        if count<=0 or not replay_by_class:
            raise ValueError('Cannot draw a replay minibatch from an empty archive')
        ids=[]
        if len(oldmemory)>=count:
            remaining={cls:list(group) for cls,group in replay_by_class.items()}
            for _ in range(count):
                classes=tuple(cls for cls,group in remaining.items() if group)
                cls=classes[int(rng.integers(0,len(classes)))]
                group=remaining[cls]
                pos=int(rng.integers(0,len(group)))
                ids.append(group.pop(pos))
                replay_sampled_class_counts[cls]+=1
        else:
            classes=tuple(replay_by_class)
            for _ in range(count):
                cls=classes[int(rng.integers(0,len(classes)))]
                group=replay_by_class[cls]
                ids.append(group[int(rng.integers(0,len(group)))])
                replay_sampled_class_counts[cls]+=1
        examples=[training_data[i] for i in ids]
        x=torch.stack([item[0] for item in examples])
        y=torch.as_tensor([int(item[1]) for item in examples],dtype=torch.long)
        record_task_replay(ids)
        return x,y,count,ids
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
            value=float((model.loss_for_labels(
                batch[:valid],labels,task_ids=known_task_ids,
                task_ids_by_example=_task_ids_for_indices(
                    subset,task_ids_by_index,known_task_ids))
                if hasattr(getattr(model,'classifier',None),'task_classes')
                else F.cross_entropy(logits,labels)).item())
        model.train(was_train)
        return value
    old_loss_before=local_old_loss() if cfg.prepost_diagnostics else None
    step_count=0; microbatch_count=0; loss_total=0.; last_projection=0.; num_violations=0;qp_calls=0
    def load_memory_microbatch(ids,tag,padding_pool=None):
        x,y=next(iter(_example_loader(training_data,ids,cfg.batch_size,
            cfg.seed+round_id+client.client_id+tag,shuffle=False)))
        valid=len(y)
        if valid<cfg.batch_size:
            missing=cfg.batch_size-valid
            if cfg.replay_padding_policy=='archive_sample' and padding_pool:
                # Excluded padding copies still affect train-mode BatchNorm. Fill
                # a short final archive chunk from the complete visible replay
                # archive instead of multiplying only its tail examples.
                pad_rng=np.random.default_rng(
                    cfg.seed+9_191_377+rng_round*1_000_003+
                    client.client_id*10_007+int(tag))
                pad_ids=pad_rng.choice(np.asarray(padding_pool,dtype=np.int64),
                    size=missing,replace=len(padding_pool)<missing).tolist()
                # Keep the padding-only crops out of the current/replay
                # augmentation RNG stream so this control changes only which
                # examples BatchNorm sees in otherwise excluded slots.
                augmentation_rng=(training_data.rng
                    if isinstance(training_data,_CIFARCropFlipDataset) else None)
                if augmentation_rng is not None:training_data.rng=pad_rng
                try:
                    padded=[training_data[int(i)] for i in pad_ids]
                finally:
                    if augmentation_rng is not None:training_data.rng=augmentation_rng
                px=torch.stack([item[0] for item in padded])
                py=torch.as_tensor([int(item[1]) for item in padded],dtype=torch.long)
                x=torch.cat((x,px),dim=0)
                y=torch.cat((y,py),dim=0)
            else:
                pad=torch.arange(missing)%valid
                x=torch.cat((x,x[pad]),dim=0)
                y=torch.cat((y,y[pad]),dim=0)
        return x,y,valid

    def replay_forward(x):
        """Forward replay chunks while optionally preserving BatchNorm buffers.

        The freeze policy keeps train-mode batch statistics for the gradient,
        but prevents every replay microbatch from advancing running moments.
        This isolates replay-buffer multiplicity from the optimizer objective.
        """
        if (cfg.method!='replay' or
                cfg.replay_bn_policy!='freeze_replay_buffers'):
            return model(x)
        saved=[]
        batchnorm_base=torch.nn.modules.batchnorm._BatchNorm
        for layer in model.modules():
            if isinstance(layer,batchnorm_base):
                for name in ('running_mean','running_var','num_batches_tracked'):
                    value=getattr(layer,name,None)
                    if value is not None:
                        saved.append((layer,name,value.detach().clone()))
        try:
            return model(x)
        finally:
            with torch.no_grad():
                for layer,name,value in saved:
                    getattr(layer,name).copy_(value)

    def replay_task_features(x):
        """Compute Task-IL replay features without an unused active-head pass."""
        if (cfg.method!='replay' or
                cfg.replay_bn_policy!='freeze_replay_buffers'):
            return model.features(x)
        saved=[]
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
                for name in ('running_mean','running_var','num_batches_tracked'):
                    value=getattr(layer,name,None)
                    if value is not None:
                        saved.append((layer,name,value.detach().clone()))
        try:
            return model.features(x)
        finally:
            with torch.no_grad():
                for layer,name,value in saved:
                    getattr(layer,name).copy_(value)

    def full_training_batches(loaded):
        x,y=loaded[:2]
        original_ids=loaded[2] if len(loaded)>2 else None
        mask_partial=(cfg.stream_sync_every_batch and
                      cfg.stream_partial_batch_policy=='mask_padding_loss')
        def emit(batch_x,batch_y,batch_ids,current_count=None,replay_count=None,replay_ids=None):
            loss_count=len(batch_y)
            if mask_partial and loss_count<cfg.batch_size:
                if loss_count<1:raise RuntimeError('Partial batch has no valid loss examples')
                pad=torch.arange(cfg.batch_size-loss_count)%loss_count
                batch_x=torch.cat((batch_x,batch_x[pad]),dim=0)
                batch_y=torch.cat((batch_y,batch_y[pad]),dim=0)
            if replay_count is not None:
                return batch_x,batch_y,batch_ids,int(current_count),int(replay_count),tuple(replay_ids)
            if current_count is not None:
                return batch_x,batch_y,batch_ids,int(current_count)
            if mask_partial:return batch_x,batch_y,batch_ids,loss_count
            return batch_x,batch_y,batch_ids
        if mask_partial and original_ids is not None:
            # Recover the valid current-stream subset by ID multiplicity. The
            # same ID can also appear in padding, so count each current index
            # only as many times as it was actually supplied in this arrival.
            remaining=Counter(map(int,indices))
            valid=[]
            for idx in original_ids.tolist():
                idx=int(idx);keep=remaining.get(idx,0)>0
                valid.append(keep)
                if keep:remaining[idx]-=1
            mask=torch.as_tensor(valid,dtype=torch.bool)
            x,y=x[mask],y[mask]
            original_ids=original_ids[mask]
        sampled_mixed=(cfg.method=='replay' and
                       cfg.replay_update_policy in {'sampled_mixed',
                                                    'sampled_mixed_accumulated'})
        if not sampled_mixed or not oldmemory or cfg.replay_ratio<=0:
            yield emit(x,y,original_ids)
            return
        # Split each incoming batch into two equal current pieces, then add
        # the same number of class-uniform Replay images to each microbatch.
        # The accumulated control combines both microbatch gradients into one
        # optimizer step; sampled_mixed steps after each microbatch.
        fresh=max(1,min(cfg.batch_size-1,
                        int(round(cfg.batch_size/(1.+cfg.replay_ratio)))))
        if (cfg.stream_sync_every_batch and
                cfg.stream_partial_batch_policy=='mask_padding_loss'):
            # Partial arrivals carry only their real current examples. Replay
            # count follows that valid count exactly; no repeated current data
            # is invented to reach a nominal microbatch size.
            for offset in range(0,len(y),fresh):
                nx,ny=x[offset:offset+fresh],y[offset:offset+fresh]
                hx,hy,_,history_ids=draw_history(len(ny))
                yield emit(torch.cat((nx,hx),dim=0),
                           torch.cat((ny,hy),dim=0),None,
                           current_count=len(ny) if cfg.learning_setting=='task_il' else None,
                           replay_count=len(hy) if cfg.learning_setting=='task_il' else None,
                           replay_ids=history_ids if cfg.learning_setting=='task_il' else None)
            return
        for offset in range(0,len(y),fresh):
            nx,ny=x[offset:offset+fresh],y[offset:offset+fresh]
            current_ids=original_ids[offset:offset+fresh] if current_head_by_index is not None else None
            if len(ny)<fresh:
                fill=torch.as_tensor(rng.integers(0,len(y),size=fresh-len(ny)),
                                     dtype=torch.long)
                nx=torch.cat((nx,x[fill]),dim=0)
                ny=torch.cat((ny,y[fill]),dim=0)
                if current_ids is not None:current_ids=torch.cat((current_ids,original_ids[fill]))
            hx,hy,_,history_ids=draw_history(cfg.batch_size-fresh)
            yield emit(torch.cat((nx,hx),dim=0),
                       torch.cat((ny,hy),dim=0),current_ids,
                       current_count=len(ny) if cfg.learning_setting=='task_il' else None,
                       replay_count=len(hy) if cfg.learning_setting=='task_il' else None,
                       replay_ids=history_ids if cfg.learning_setting=='task_il' else None)

    for epoch in range(cfg.epochs):
        online_der=cfg.method in {'der','derpp'}
        keep_padding_mask=(cfg.stream_sync_every_batch and
            cfg.stream_partial_batch_policy=='mask_padding_loss')
        for loaded in _example_loader(training_data,indices,cfg.batch_size,
                                   cfg.seed+rng_round*100_003+client.client_id*127+epoch,
                                   include_indices=(online_der or keep_padding_mask or current_head_by_index is not None),
                                   full_batch=True,padding_indices=padding_indices):
            for micro_index,packed in enumerate(full_training_batches(loaded)):
                replay_example_count=0
                replay_sample_ids=()
                if len(packed)==6:
                    x,y,original_ids,loss_example_count,replay_example_count,replay_sample_ids=packed
                elif len(packed)==5:
                    x,y,original_ids,loss_example_count,replay_example_count=packed
                elif len(packed)==4:
                    x,y,original_ids,loss_example_count=packed
                else:
                    x,y,original_ids=packed
                    loss_example_count=len(y)
                accumulate_optimizer_step=(
                    cfg.method=='replay' and
                    cfg.replay_update_policy=='sampled_mixed_accumulated' and
                    bool(oldmemory) and cfg.replay_ratio>0)
                if not accumulate_optimizer_step or micro_index==0:
                    optimizer.zero_grad(set_to_none=True)
                gradient_accumulation_scale=(0.5 if accumulate_optimizer_step else 1.0)
                logical_size=len(y)
                macro_loss=0.
                si_task_grads=None
                captured_logits={}
                if logical_size!=cfg.batch_size:
                    raise RuntimeError('Training forward does not have exactly 64 images')
                if loss_example_count<1 or loss_example_count>logical_size:
                    raise RuntimeError('Training forward has an invalid loss-example count')
                if cfg.method in {'native_glfc','native_lga'} and cfg.dataset=='cifar100':
                    from .upstream_native_augmentation import original_glfc_lga_augment_batch
                    x=original_glfc_lga_augment_batch(cfg.method,x.cpu(),seed=(
                        cfg.seed+client.client_id*1_000_003+rng_round*100_003+
                        client.local_experience*1009+epoch*173+step_count))
                if cfg.method=='native_feat' and cfg.dataset=='cifar100':
                    # Exact original FEAT CIFAR-100 crop/flip/brightness
                    # composed source transformations, not generic augmentation.
                    # Only TRAIN/replay inputs; never validation or TEST.
                    from .upstream_feat_augmentation import original_feat_augment_batch
                    x=original_feat_augment_batch(x.cpu(),seed=(
                        cfg.seed+client.client_id*1_000_003+rng_round*100_003+
                        client.local_experience*1009+epoch*173+step_count))
                x=x.to(device,non_blocking=True)
                y=y.to(device,non_blocking=True)
                if cfg.method in NATIVE_METHODS and cfg.learning_setting!='task_il':
                    loss,logits=native_loss(cfg.method,model,x,y,seen_classes,
                                            teacher=native_teacher,old_classes=native_old_classes)
                    task_features=None
                elif cfg.learning_setting=='task_il':
                    task_features=model.features(x)
                    logits=model.classifier(task_features)
                else:
                    task_features=None
                    logits=model(x)
                if original_ids is not None and current_head_by_index is None:
                    for idx,z in zip(original_ids.tolist(),logits.detach().cpu()):
                        captured_logits[int(idx)]=z.clone()
                objective_logits=logits[:loss_example_count]
                objective_y=y[:loss_example_count]
                if cfg.method not in NATIVE_METHODS or cfg.learning_setting=='task_il':
                    if cfg.learning_setting=='task_il' and current_head_by_index is not None:
                        if original_ids is None or len(original_ids)!=loss_example_count:
                            raise RuntimeError('Pooled Task-IL lost fresh sample routing')
                        sources=[(current_head_by_index[int(i)],) for i in original_ids.tolist()]
                        loss=model.loss_for_labels(x[:loss_example_count],objective_y,
                            task_ids=known_task_ids,task_ids_by_example=sources,
                            features=task_features[:loss_example_count])
                    elif cfg.learning_setting=='task_il':
                        if task_classes is None or not set(objective_y.tolist()).issubset(set(task_classes)):
                            raise RuntimeError('Task-IL labels do not match the known local task')
                        loss=task_head_loss(objective_logits,objective_y,task_classes)
                    elif active_pair_classes is not None:
                        # ER-ACE is intentionally asymmetric in its source paper:
                        # mask historical logits for *new* examples only, unlike
                        # standard ER, EWC, SI, MAS, GEM, LwF and DER.
                        current_competition=(active_pair_classes if cfg.method=='er_ace'
                                             else seen_classes)
                        loss=canonical_active_loss(objective_logits,objective_y,current_competition)
                    elif cfg.learning_setting=='task_il':
                        if task_classes is None or not set(objective_y.tolist()).issubset(set(task_classes)):
                            raise RuntimeError('Task-IL labels do not match the known local task')
                        task_mask=torch.ones(logits.shape[1],dtype=torch.bool,device=device)
                        task_mask[torch.as_tensor(task_classes,dtype=torch.long,device=device)]=False
                        logits=logits.masked_fill(task_mask.unsqueeze(0),-1e9)
                    if cfg.learning_setting!='task_il' and active_pair_classes is None and cfg.method=='er_ace':
                        # ER-ACE stream loss: mask historical classes absent from the
                        # *current* physical batch, but retain future/unseen output columns.
                        mask=torch.zeros(logits.shape[1],dtype=torch.bool,device=device)
                        current_labels=set(int(v) for v in objective_y.unique().tolist())
                        for past_cls in client.observed_classes:
                            if past_cls not in current_labels and past_cls < logits.shape[1]:
                                mask[past_cls]=True
                        stream_logits=objective_logits.masked_fill(mask.unsqueeze(0),-1e9)
                        loss=F.cross_entropy(stream_logits,objective_y)
                    elif cfg.learning_setting!='task_il' and active_pair_classes is None:
                        loss=F.cross_entropy(objective_logits,objective_y)
                if (cfg.learning_setting=='task_il' and replay_example_count>0
                        and cfg.method=='replay'
                        and cfg.replay_update_policy in {'sampled_mixed','sampled_mixed_accumulated'}):
                    replay_start=loss_example_count
                    replay_end=replay_start+replay_example_count
                    replay_loss=model.loss_for_labels(x[replay_start:replay_end],
                        y[replay_start:replay_end],task_ids=known_task_ids,
                        task_ids_by_example=_task_ids_for_indices(
                            replay_sample_ids,task_ids_by_index,known_task_ids),
                        features=task_features[replay_start:replay_end])
                    valid_count=loss_example_count+replay_example_count
                    current_weight=loss_example_count/valid_count
                    replay_weight=replay_example_count/valid_count
                    loss=loss*current_weight+replay_loss*replay_weight
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
                exhaustive_method_replay=(cfg.method in {'er_ace','der','derpp'}
                                          and bool(oldmemory))
                if cfg.method=='lwf' and teacher is not None:
                    temperature=2.0
                    if cfg.learning_setting=='task_il':
                        old_task_ids=known_task_ids-{task_head_key(task_id)}
                        if old_task_ids:
                            with torch.no_grad():teacher_features=teacher.features(x)
                            distillation=[]
                            for task_id in sorted(old_task_ids):
                                student=model.classifier(task_features,task_id)
                                past=teacher.classifier(teacher_features,task_id)
                                distillation.append(F.kl_div(
                                    F.log_softmax(student/temperature,dim=1),
                                    F.softmax(past/temperature,dim=1),
                                    reduction='batchmean'))
                            loss=loss+cfg.der_alpha*temperature**2*torch.stack(distillation).mean()
                    else:
                        with torch.no_grad():t_logits=teacher(x)
                        old_labels=sorted(c for c,s in client.first_seen_local_experience.items()
                                          if s < client.local_experience)
                        if old_labels:
                            ids=torch.tensor(old_labels,dtype=torch.long,device=device)
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
                primary_replay_objective=(
                    cfg.method=='replay' and bool(oldmemory) and cfg.replay_ratio>0
                    and cfg.replay_update_policy in {'exhaustive','sampled_separate'})
                exhaustive_replay=(
                    cfg.method in ({'replay'}|NATIVE_METHODS)
                    and bool(oldmemory) and cfg.replay_ratio>0
                    and not (cfg.method=='replay' and
                             cfg.replay_update_policy!='exhaustive'))
                if exhaustive_replay or primary_replay_objective:
                    # Mix the mean incoming-batch objective and the mean over
                    # every stored exemplar. Chunk scaling makes this exactly
                    # the full-buffer mean while keeping accelerator memory
                    # bounded by cfg.batch_size.
                    loss=loss/(1.+cfg.replay_ratio)
                macro_loss+=float(loss.detach().item())*(len(y)/logical_size)
                (loss*(len(y)/logical_size)*gradient_accumulation_scale).backward()
                if exhaustive_method_replay:
                    for start in range(0,len(oldmemory),cfg.batch_size):
                        ids=oldmemory[start:start+cfg.batch_size]
                        hx,hy,valid=load_memory_microbatch(ids,step_count+start,padding_pool=oldmemory)
                        hx,hy=hx.to(device,non_blocking=True),hy.to(device,non_blocking=True)
                        scale=cfg.replay_ratio*valid/len(oldmemory)
                        hy=hy[:valid]
                        if cfg.learning_setting=='task_il':
                            replay_features=replay_task_features(hx)[:valid]
                            source_task_ids=_task_ids_for_indices(
                                ids[:valid],task_ids_by_index,known_task_ids)
                            supervised=model.loss_for_labels(
                                hx[:valid],hy,task_ids=known_task_ids,
                                task_ids_by_example=source_task_ids,
                                features=replay_features)
                            if cfg.method in {'der','derpp'}:
                                if any(len(task_ids)!=1 for task_ids in source_task_ids):
                                    raise ValueError('Task-IL DER requires one source task per replay sample')
                                replay_logits=torch.cat([
                                    model.classifier(replay_features[j:j+1],task_ids[0])
                                    for j,task_ids in enumerate(source_task_ids)],dim=0)
                                positions=[j for j,index in enumerate(ids[:valid])
                                           if index in client.memory_logits]
                                terms=[]
                                if positions:
                                    losses=[]
                                    for j in positions:
                                        target=client.memory_logits[ids[j]].to(device)
                                        if target.shape!=replay_logits[j].shape:
                                            raise RuntimeError('Task-IL DER source-head logits changed shape')
                                        losses.append(F.mse_loss(replay_logits[j],target))
                                    terms.append(cfg.der_alpha*torch.stack(losses).mean())
                                if cfg.method=='derpp':
                                    terms.append(cfg.der_beta*supervised)
                                elif len(positions)<valid:
                                    missing=[j for j in range(valid) if j not in set(positions)]
                                    terms.append(model.loss_for_labels(
                                        hx[missing],hy[missing],task_ids=known_task_ids,
                                        task_ids_by_example=[source_task_ids[j] for j in missing],
                                        features=replay_features[missing]))
                                replay_loss=sum(terms,torch.zeros((),device=device))
                            else:
                                replay_loss=supervised
                            record_task_replay(ids[:valid],scale)
                            if cfg.method=='er_ace':
                                replay_loss=cfg.der_beta*replay_loss
                        elif cfg.method=='er_ace':
                            replay_logits=replay_forward(hx)[:valid]
                            replay_loss=(canonical_active_loss(replay_logits,hy,seen_classes)
                                if active_pair_classes is not None else F.cross_entropy(replay_logits,hy))
                            replay_loss=cfg.der_beta*replay_loss
                        else:
                            replay_logits=replay_forward(hx)[:valid]
                            terms=[]
                            positions=[j for j,i in enumerate(ids) if i in client.memory_logits]
                            if positions:
                                old_logits=torch.stack([client.memory_logits[ids[j]]
                                                        for j in positions]).to(device)
                                if old_logits.shape!=replay_logits[positions].shape:
                                    raise RuntimeError('DER logits incompatible with classifier schema')
                                terms.append(cfg.der_alpha*F.mse_loss(
                                    replay_logits[positions],old_logits))
                            if cfg.method=='derpp':
                                supervised=(canonical_active_loss(replay_logits,hy,seen_classes)
                                    if active_pair_classes is not None
                                    else F.cross_entropy(replay_logits,hy))
                                terms.append(cfg.der_beta*supervised)
                            elif len(positions)<valid:
                                missing=[j for j in range(valid) if j not in set(positions)]
                                terms.append(F.cross_entropy(replay_logits[missing],hy[missing]))
                            replay_loss=sum(terms,torch.zeros((),device=device))
                        (replay_loss*scale).backward()
                if cfg.method=='replay' and bool(oldmemory) and cfg.replay_ratio>0 \
                        and cfg.replay_update_policy=='sampled_separate':
                    # One optimizer step per current physical batch, but draw
                    # only one replay minibatch. This isolates replay coverage
                    # from the V21 two-step batch partition.
                    replay_ids=rng.choice(
                        oldmemory,size=cfg.batch_size//2,
                        replace=len(oldmemory)<cfg.batch_size//2).tolist()
                    hx,hy,valid=load_memory_microbatch(replay_ids,3_000_000+step_count)
                    hx,hy=hx.to(device,non_blocking=True),hy.to(device,non_blocking=True)
                    history_scale=cfg.replay_ratio/(1.+cfg.replay_ratio)
                    if cfg.learning_setting=='task_il':
                        replay_features=replay_task_features(hx)[:valid]
                        history_loss=model.loss_for_labels(
                                      hx[:valid],hy[:valid],task_ids=known_task_ids,
                                      task_ids_by_example=_task_ids_for_indices(
                                          replay_ids[:valid],task_ids_by_index,known_task_ids),
                                      features=replay_features)
                    else:
                        replay_logits=replay_forward(hx)[:valid]
                        history_loss=(canonical_active_loss(replay_logits,hy[:valid],seen_classes)
                                      if active_pair_classes is not None else
                                      F.cross_entropy(replay_logits,hy[:valid]))
                    record_task_replay(replay_ids[:valid],history_scale)
                    (history_loss*history_scale).backward()
                    macro_loss+=float(history_loss.detach().item())*history_scale
                if exhaustive_replay:
                    memory_scale=cfg.replay_ratio/(1.+cfg.replay_ratio)
                    replay_ids=(list(map(int,replay_order_rng.permutation(oldmemory)))
                                if (cfg.method=='replay' and
                                    cfg.replay_buffer_order_policy=='shuffled_each_pass')
                                else oldmemory)
                    for start in range(0,len(replay_ids),cfg.batch_size):
                        ids=replay_ids[start:start+cfg.batch_size]
                        hx,hy,valid=load_memory_microbatch(ids,step_count+start,padding_pool=oldmemory)
                        hx,hy=hx.to(device,non_blocking=True),hy.to(device,non_blocking=True)
                        if cfg.method in NATIVE_METHODS and cfg.learning_setting!='task_il':
                            history_loss,_=native_loss(cfg.method,model,hx[:valid],hy[:valid],seen_classes,
                                teacher=native_teacher,old_classes=native_old_classes)
                        elif cfg.learning_setting=='task_il':
                            replay_features=replay_task_features(hx)[:valid]
                            history_loss=model.loss_for_labels(
                                hx[:valid],hy[:valid],task_ids=known_task_ids,
                                task_ids_by_example=_task_ids_for_indices(
                                    ids[:valid],task_ids_by_index,known_task_ids),
                                features=replay_features)
                        else:
                            history_logits=replay_forward(hx)[:valid]
                            history_loss=(canonical_active_loss(history_logits,hy[:valid],seen_classes)
                                if active_pair_classes is not None else
                                F.cross_entropy(history_logits,hy[:valid]))
                        chunk_weight=memory_scale*valid/len(oldmemory)
                        record_task_replay(ids[:valid],chunk_weight)
                        (history_loss*chunk_weight).backward()
                        macro_loss+=float(history_loss.detach().item())*chunk_weight
                if cfg.method in {'gem','agem'} and oldmemory:
                    if active_pair_classes is not None:
                        freeze_other_class_rows(model,seen_classes)
                    g=_flatten_grads(params)
                    if cfg.method=='agem':
                        # A-GEM's reference direction is the mean gradient of
                        # the complete retained buffer, evaluated in bounded
                        # batches rather than sampled from it.
                        refidx=oldmemory
                        ref=_reference_gradient(model,params,training_data,refidx,cfg.batch_size,device,
                                                cfg.seed+round_id+client.client_id,seen_classes,
                                                task_ids=known_task_ids,
                                                task_ids_by_index=task_ids_by_index)
                        z,last_projection,v=agem_project(g,ref)
                        if v:record_task_replay(refidx,1.)
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
                                                                historical_classes,known_task_ids,
                                                                task_ids_by_index))
                        z,last_projection,v=gem_project(g,refs)
                        if v:
                            for mem in groups.values():
                                if mem:record_task_replay(mem,1.)
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
                should_step_optimizer=(
                    not accumulate_optimizer_step or micro_index==1)
                if should_step_optimizer:
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
                if should_step_optimizer:
                    step_count+=1
                microbatch_count+=1
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
            seed=cfg.seed+rng_round*100003+client.client_id*127+client.local_experience,
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
    if cfg.method=='native_feat' and experience_end:
        from .upstream_feat_replay import source_feat_select_local
        source_feat_select_local(model,client,bank,cfg,device,indices,
                                 active_pair_classes or ())
    if cfg.method in ({'gem','agem','replay','er_ace'}):
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
    if cfg.method in ({'replay','er_ace','der','derpp','gem','agem'}|NATIVE_METHODS):
        client.trained_classes.update(int(bank.train.targets[i]) for i in oldmemory)
    class_counts=dict(Counter(int(bank.train.targets[i]) for i in indices))
    if active_pair_classes is not None:
        # Legacy row-wise fusion records every activated class. Task-IL server
        # head exposure is tracked separately by task ID below.
        client.activated_labels.update(seen_classes)
        class_update_steps={int(c):step_count for c in seen_classes}
    else:
        client.activated_labels.update(class_counts)
        class_update_steps={int(c):step_count for c in class_counts}

    task_head_examples={}
    if cfg.learning_setting=='task_il' and hasattr(getattr(model,'classifier',None),'task_classes'):
        from .models import task_head_key
        if current_head_by_index is not None:
            for key,count in Counter(current_head_by_index.values()).items():
                task_head_examples[key]=float(count*cfg.epochs)
        elif task_classes:
            active_key=task_head_key(task_id)
            task_head_examples[active_key]=float(len(indices)*cfg.epochs)
        for key,value in replay_task_head_exposure.items():
            task_head_examples[key]=task_head_examples.get(key,0.)+float(value)
    return {'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'examples':len(indices),'class_counts':class_counts,'class_update_steps':class_update_steps,
            'task_head_examples':task_head_examples,
            'active_pair_classes':list(active_pair_classes) if active_pair_classes else None,'steps':step_count,'loss':loss_total/max(1,microbatch_count),
            'constraint_violations':num_violations,'qp_calls':qp_calls,
            'projection_norm_last':last_projection,'memory_samples':sum(map(len,client.memory.values())),
            'replay_sampled_class_counts':{int(k):int(v) for k,v in sorted(replay_sampled_class_counts.items())},
            'replay_sampled_examples':int(sum(replay_sampled_class_counts.values())),
            'method_used':cfg.method,'fedprox_mu':cfg.fedprox_mu,
            'local_old_loss_before':old_loss_before,
            'local_old_loss_after':old_loss_after,
            'optimizer_state_policy':cfg.optimizer_state_policy,
            'sgd_momentum':cfg.momentum,'sgd_weight_decay':cfg.weight_decay,
            'classification_competitor_classes':list(seen_classes)}
