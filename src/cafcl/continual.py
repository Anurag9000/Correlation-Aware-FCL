"""Local SGD, ER, GEM, A-GEM and a diagonal EWC baseline.

Client memory stores *training indices*, never copies of held-out test data.
GEM uses its actual dual QP and verifies primal constraints. Inaccurate
projections cause the job to fail instead of silently defaulting to SGD.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,field
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from scipy.optimize import minimize

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
    received_memory:dict[int,list[int]]=field(default_factory=dict)
    received_example_payloads:dict[int,tuple]=field(default_factory=dict)
    received_gradient_supports:dict[int,tuple[int,...]]=field(default_factory=dict)
    pending_importance:dict[str,dict]=field(default_factory=dict)
    federated_cl_version:int=0
    lwf_teacher_from_server:bool=False
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
    def update_memory(self,indices,targets,budget,rng,policy='random_per_class',*,total_limit=0):
        """Private 1000-example replay budget; federated archive is not capped."""
        from .federated_cl import combined_memory
        by_class=defaultdict(list)
        for idx in indices:by_class[int(targets[idx])].append(int(idx))
        if total_limit>0:
            merged=combined_memory(self.private_memory,by_class)
            samples=[(c,i) for c,ids in sorted(merged.items()) for i in ids]
            if len(samples)>total_limit:
                chosen=sorted(int(i) for i in rng.choice(len(samples),size=total_limit,replace=False))
                samples=[samples[i] for i in chosen]
            private=defaultdict(list)
            for c,i in samples:private[c].append(i)
            self.private_memory=dict(private)
        elif budget>0:
            for cls,samples in by_class.items():
                old=self.private_memory.get(cls,[])
                pool=list(dict.fromkeys(old+samples))
                if len(pool)>budget:
                    if policy=='recent_per_class':pool=pool[-budget:]
                    else:
                        chosen=rng.choice(len(pool),size=budget,replace=False)
                        pool=[pool[int(i)] for i in sorted(chosen)]
                self.private_memory[cls]=pool
        self.memory=combined_memory(self.private_memory,self.received_memory)
        retained={int(i) for group in self.memory.values() for i in group}
        for i in tuple(self.memory_logits):
            if i not in retained:del self.memory_logits[i]



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


def _example_loader(dataset,indices,batch_size,seed,shuffle=True):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset,list(indices)),batch_size=batch_size,shuffle=shuffle,
                      num_workers=0,generator=generator,pin_memory=torch.cuda.is_available())


def _reference_gradient(model,params,dataset,indices,batch_size,device,seed,active_pair_classes=None):
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
        # Logical reference gradient: weighted accumulation over bounded physical
        # batches, without materializing 1024 images/backbone activations on the GPU.
        total=min(len(indices),batch_size)
        micro=min(batch_size,16)
        order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed))[:total]
        selected=[indices[int(i)] for i in order]
        for x,y in _example_loader(dataset,selected,micro,seed,shuffle=False):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            (F.cross_entropy(model(x),y)*(len(y)/total)).backward()
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
    if client.shared_weights is None:
        raise RuntimeError('Client must synchronize before local training')
    model.load_state_dict(client.shared_weights)
    model.to(device)
    model.train()
    if active_pair_classes is not None:
        from .class_indexed import canonical_active_loss,freeze_other_class_rows
        active_pair_classes=tuple(sorted(set(int(c) for c in active_pair_classes)))
        if len(active_pair_classes)<2:
            raise ValueError('Local task requires at least two distinct labels')
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
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum)
    if cfg.optimizer_state_policy=='persist' and client.optimizer_state is not None:
        optimizer.load_state_dict(client.optimizer_state)
    params=_trainable_params(model)
    if cfg.method=='si':
        if client.si_omega is None:
            client.si_omega={n:torch.zeros_like(p.detach().cpu())
                             for n,p in model.named_parameters()}
            client.si_path={n:torch.zeros_like(p.detach().cpu())
                            for n,p in model.named_parameters()}
            client.si_displacement={n:torch.zeros_like(p.detach().cpu())
                                    for n,p in model.named_parameters()}
        if client.si_stage>=0 and client.local_experience>client.si_stage:
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
    def local_old_loss():
        if not oldmemory:return None
        was_train=model.training
        model.eval()
        with torch.no_grad():
            subset=oldmemory[:min(cfg.micro_batch_size,len(oldmemory))]
            batch=torch.stack([training_data[i][0] for i in subset]).to(device)
            labels=torch.tensor([bank.train.targets[i] for i in subset],device=device)
            value=float(F.cross_entropy(model(batch),labels).item())
        model.train(was_train)
        return value
    old_loss_before=local_old_loss() if cfg.prepost_diagnostics else None
    step_count=0; loss_total=0.; last_projection=0.; num_violations=0;qp_calls=0
    for epoch in range(cfg.epochs):
        if cfg.method=='replay' and oldmemory and active_pair_classes is None:
            replay_count=min(len(oldmemory),int(round(len(indices)*cfg.replay_ratio)))
            chosen=rng.choice(oldmemory,size=replay_count,replace=len(oldmemory)<replay_count).tolist()
            batch_indices=indices+chosen
        else:
            batch_indices=indices
        for x,y in _example_loader(training_data,batch_indices,cfg.batch_size,
                                   cfg.seed+round_id*100_003+client.client_id*127+epoch):
            optimizer.zero_grad(set_to_none=True)
            logical_size=len(y)
            macro_loss=0.
            for x,y in zip(x.split(cfg.micro_batch_size), y.split(cfg.micro_batch_size)):
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                logits=model(x)
                if active_pair_classes is not None:
                    loss=canonical_active_loss(logits,y,active_pair_classes)
                elif cfg.learning_setting=='task_il':
                    if task_classes is None or not set(y.tolist()).issubset(set(task_classes)):
                        raise RuntimeError('Task-IL labels do not match the known local task')
                    task_mask=torch.ones(logits.shape[1],dtype=torch.bool,device=device)
                    task_mask[torch.as_tensor(task_classes,dtype=torch.long,device=device)]=False
                    logits=logits.masked_fill(task_mask.unsqueeze(0),-1e9)
                if active_pair_classes is None and cfg.method=='er_ace':
                    # ER-ACE stream loss: mask historical classes absent from the
                    # *current* minibatch, but retain future/unseen output columns.
                    mask=torch.zeros(logits.shape[1],dtype=torch.bool,device=device)
                    current_labels=set(int(v) for v in y.unique().tolist())
                    for past_cls in client.observed_classes:
                        if past_cls not in current_labels and past_cls < logits.shape[1]:
                            mask[past_cls]=True
                    stream_logits=logits.masked_fill(mask.unsqueeze(0),-1e9)
                    loss=F.cross_entropy(stream_logits,y)
                elif active_pair_classes is None:
                    loss=F.cross_entropy(logits,y)
                if cfg.method=='replay' and oldmemory and active_pair_classes is not None:
                    ref_count=min(len(oldmemory),cfg.micro_batch_size,max(1,int(round(len(y)*cfg.replay_ratio))))
                    chosen=rng.choice(oldmemory,size=ref_count,replace=False).tolist()
                    rx,ry=next(iter(_example_loader(training_data,chosen,ref_count,
                        cfg.seed+round_id+client.client_id,shuffle=False)))
                    loss=loss+F.cross_entropy(model(rx.to(device)),ry.to(device))
                if cfg.method in {'er_ace','der','derpp'} and oldmemory:
                    ref_count=min(len(oldmemory),cfg.micro_batch_size,max(1,int(round(len(y)*cfg.replay_ratio))))
                    chosen=rng.choice(oldmemory,size=ref_count,replace=False).tolist()
                    rx,ry=next(iter(_example_loader(training_data,chosen,ref_count,
                                                   cfg.seed+round_id+client.client_id,shuffle=False)))
                    rx,ry=rx.to(device,non_blocking=True),ry.to(device,non_blocking=True)
                    replay_logits=model(rx)
                    if cfg.method=='er_ace':
                        loss=loss+cfg.der_beta*F.cross_entropy(replay_logits,ry)
                    else:
                        missing=[int(i) for i in chosen if int(i) not in client.memory_logits]
                        if missing:
                            raise RuntimeError(f'DER memory logits missing for {missing[:3]}')
                        old_logits=torch.stack([client.memory_logits[int(i)] for i in chosen]).to(device)
                        if old_logits.shape!=replay_logits.shape:
                            raise RuntimeError('DER logits incompatible with classifier schema')
                        loss=loss+cfg.der_alpha*F.mse_loss(replay_logits,old_logits)
                        if cfg.method=='derpp':
                            loss=loss+cfg.der_beta*F.cross_entropy(replay_logits,ry)
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
                    freeze_other_class_rows(model,active_pair_classes)
                g=_flatten_grads(params)
                if cfg.method=='agem':
                    refidx=rng.choice(oldmemory,size=min(cfg.batch_size,len(oldmemory)),replace=False).tolist()
                    ref=_reference_gradient(model,params,training_data,refidx,cfg.batch_size,device,
                                            cfg.seed+round_id+client.client_id,active_pair_classes)
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
                                groups[client.sample_first_stage.get(idx,client.first_seen_stage[cls])].append(idx)
                    else:
                        groups={0:oldmemory}
                    for group in sorted(groups):
                        mem=groups[group]
                        if mem:
                            refs.append(_reference_gradient(model,params,training_data,mem,
                                                            cfg.batch_size,device,cfg.seed+group,active_pair_classes))
                    z,last_projection,v=gem_project(g,refs)
                    qp_calls+=int(v>0)
                num_violations+=v
                _set_grads(params,z)
            if active_pair_classes is not None:
                freeze_other_class_rows(model,active_pair_classes)
            frozen=None
            if active_pair_classes is not None:
                from .class_indexed import snapshot_frozen_rows,restore_frozen_rows
                frozen=snapshot_frozen_rows(model,active_pair_classes)
            if cfg.method=='si':
                before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                applied_grad={n:(p.grad.detach().cpu().clone() if p.grad is not None
                                 else torch.zeros_like(p.detach().cpu()))
                              for n,p in model.named_parameters()}
            optimizer.step()
            if frozen is not None:
                restore_frozen_rows(model,optimizer,frozen)
            if cfg.method=='si':
                for n,p in model.named_parameters():
                    displacement=p.detach().cpu()-before[n]
                    client.si_path[n]+=-applied_grad[n]*displacement
                    client.si_displacement[n]+=displacement
            step_count+=1
            loss_total+=macro_loss
    # Diagonal Fisher approximation for EWC, consolidated at each stage transition.
    if cfg.method=='ewc' and (experience_end or (round_id+1)%cfg.rounds_per_stage==0):
        if client.ewc_anchor is None or client.local_experience > client.ewc_stage:
            # Use a bounded local batch; this is an approximation, recorded in diagnostics.
            x,y=next(iter(_example_loader(training_data,indices,min(cfg.micro_batch_size,128),
                                          cfg.seed+round_id+client.client_id)))
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(device)),y.to(device)).backward()
            fresh_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            fresh_fisher={n:(p.grad.detach().cpu().square() if p.grad is not None else torch.zeros_like(p.cpu())) for n,p in model.named_parameters()}
            from .federated_cl import add_importance,weighted_importance
            add_importance(client,'ewc',fresh_fisher,fresh_anchor)
            client.ewc_fisher,client.ewc_anchor=weighted_importance(
                client.ewc_fisher,client.ewc_anchor,fresh_fisher,fresh_anchor)
            client.ewc_stage=client.local_experience
    if (cfg.method=='mas' and (experience_end or (round_id+1)%cfg.rounds_per_stage==0)
            and client.local_experience > client.mas_stage):
        # MAS surrogate importance uses output sensitivity, independent of labels.
        xm,_=next(iter(_example_loader(training_data,indices,min(cfg.micro_batch_size,128),
                                      cfg.seed+round_id+client.client_id)))
        model.zero_grad(set_to_none=True)
        squared_output=model(xm.to(device)).square().sum(dim=1).mean()
        squared_output.backward()
        omega={n:(p.grad.detach().cpu().abs() if p.grad is not None else
                   torch.zeros_like(p.detach().cpu())) for n,p in model.named_parameters()}
        fresh_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        from .federated_cl import add_importance,weighted_importance
        add_importance(client,'mas',omega,fresh_anchor)
        client.mas_omega,client.mas_anchor=weighted_importance(
            client.mas_omega,client.mas_anchor,omega,fresh_anchor)
        client.mas_stage=client.local_experience
    # Never allocate replay/GEM memory for non-memory algorithms.
    if cfg.method in {'gem','agem','replay','er_ace','der','derpp'}:
        client.update_memory(indices,bank.train.targets,cfg.memory_per_class,rng,cfg.memory_policy,
                             total_limit=(cfg.replay_buffer_capacity if cfg.method=='replay' else 0))
    if cfg.method in {'der','derpp'} and client.memory:
        # Logits recorded at memory insertion/consolidation, never from test data.
        # This is a stage-consolidated local DER adaptation rather than the
        # paper's step-wise online reservoir update.
        retained=[int(i) for group in client.memory.values() for i in group]
        unseen=[i for i in retained if i not in client.memory_logits]
        if unseen:
            model.eval()
            with torch.no_grad():
                for batch_start in range(0,len(unseen),cfg.micro_batch_size):
                    batch_ids=unseen[batch_start:batch_start+cfg.micro_batch_size]
                    x=torch.stack([training_data[i][0] for i in batch_ids]).to(device)
                    logits=model(x).detach().cpu()
                    for i,z in zip(batch_ids,logits):
                        client.memory_logits[int(i)]=z.clone()
            model.train()
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
    from collections import Counter
    class_counts=dict(Counter(int(bank.train.targets[i]) for i in indices))
    if active_pair_classes is not None:
        client.activated_labels.update(active_pair_classes)
        class_update_steps={int(c):step_count for c in active_pair_classes}
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
            'optimizer_state_policy':cfg.optimizer_state_policy}
