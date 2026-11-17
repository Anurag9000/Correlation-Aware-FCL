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
    memory_logits:dict[int,torch.Tensor]=field(default_factory=dict)
    mas_omega:dict|None=None
    mas_anchor:dict|None=None
    lwf_teacher:dict|None=None
    lwf_stage:int=-1
    ewc_stage:int=-1
    def receive(self,weights,version):
        self.shared_weights={k:v.detach().cpu().clone() for k,v in weights.items()}
        self.model_version=version
    def arrival(self,indices,labels,stage):
        if stage > self.current_stage:
            self.current_indices=[]
            self.current_stage=stage
        self.current_indices.extend(indices)
        self.seen_indices.update(indices)
        self.observed_classes.update(labels)
        for cls in labels:
            self.first_seen_stage.setdefault(int(cls),int(stage))
    def update_memory(self,indices,targets,budget, rng, policy='random_per_class'):
        if budget <= 0:return
        by_class=defaultdict(list)
        for idx in indices: by_class[int(targets[idx])].append(int(idx))
        for cls,samples in by_class.items():
            old=self.memory.get(cls,[])
            pool=list(dict.fromkeys(old+samples))
            if len(pool)>budget:
                if policy == 'recent_per_class':
                    pool=pool[-budget:]
                else:
                    keep=rng.choice(len(pool),size=budget,replace=False)
                    pool=[pool[int(j)] for j in sorted(keep)]
            self.memory[cls]=pool
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


def _example_loader(dataset,indices,batch_size,seed,shuffle=True):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset,list(indices)),batch_size=batch_size,shuffle=shuffle,
                      num_workers=0,generator=generator,pin_memory=torch.cuda.is_available())


def _reference_gradient(model,params,dataset,indices,batch_size,device,seed):
    model.zero_grad(set_to_none=True)
    batch=next(iter(_example_loader(dataset,indices,batch_size,seed)))
    x,y=batch
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
    loss=F.cross_entropy(model(x),y)
    loss.backward()
    return _flatten_grads(params)


def local_train(model,client,bank,cfg,device,round_id):
    """The global model is loaded before this function; client weights stay CPU-side."""
    if client.shared_weights is None:
        raise RuntimeError('Client must synchronize before local training')
    model.load_state_dict(client.shared_weights)
    model.to(device)
    model.train()
    indices=list(dict.fromkeys(client.current_indices))
    if not indices:
        return None
    rng=np.random.default_rng(cfg.seed+round_id*137+client.client_id*100003)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg.lr)
    params=_trainable_params(model)
    # Persist an earlier global teacher across all rounds of the new local stage.
    if cfg.method=='lwf' and client.current_stage>client.lwf_stage:
        client.lwf_teacher={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} \
            if client.lwf_stage>=0 else None
        client.lwf_stage=client.current_stage
    teacher=None
    if cfg.method=='lwf' and client.lwf_teacher is not None:
        import copy
        teacher=copy.deepcopy(model)
        teacher.load_state_dict(client.lwf_teacher)
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    anchor={name:p.detach().clone() for name,p in model.named_parameters()}
    # FedProx is a local objective, not a different aggregation formula.

    # Rehearsal and constraint memory are built only from previous locally trained examples.
    eligible_history={cls:inds for cls,inds in client.memory.items()
                      if client.first_seen_stage.get(cls,client.current_stage)<client.current_stage}
    oldmemory=[i for group in eligible_history.values() for i in group]
    step_count=0; loss_total=0.; last_projection=0.; num_violations=0;qp_calls=0
    for epoch in range(cfg.epochs):
        if cfg.method=='replay' and oldmemory:
            replay_count=min(len(oldmemory),int(round(len(indices)*cfg.replay_ratio)))
            chosen=rng.choice(oldmemory,size=replay_count,replace=len(oldmemory)<replay_count).tolist()
            batch_indices=indices+chosen
        else:
            batch_indices=indices
        for x,y in _example_loader(bank.train,batch_indices,cfg.batch_size,
                                   cfg.seed+round_id*100_003+client.client_id*127+epoch):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            if cfg.method=='er_ace':
                # ER-ACE stream loss: mask historical classes absent from the
                # *current* minibatch, but retain future/unseen output columns.
                mask=torch.zeros(logits.shape[1],dtype=torch.bool,device=device)
                current_labels=set(int(v) for v in y.unique().tolist())
                for past_cls in client.observed_classes:
                    if past_cls not in current_labels and past_cls < logits.shape[1]:
                        mask[past_cls]=True
                stream_logits=logits.masked_fill(mask.unsqueeze(0),-1e9)
                loss=F.cross_entropy(stream_logits,y)
            else:
                loss=F.cross_entropy(logits,y)
            if cfg.method in {'er_ace','der','derpp'} and oldmemory:
                ref_count=min(len(oldmemory),max(1,int(round(len(y)*cfg.replay_ratio))))
                chosen=rng.choice(oldmemory,size=ref_count,replace=False).tolist()
                rx,ry=next(iter(_example_loader(bank.train,chosen,ref_count,
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
                old_labels=sorted(c for c,s in client.first_seen_stage.items()
                                  if s < client.current_stage)
                if old_labels:
                    ids=torch.tensor(old_labels,dtype=torch.long,device=device)
                    temperature=2.0
                    student=F.log_softmax(logits.index_select(1,ids)/temperature,dim=1)
                    past=F.softmax(t_logits.index_select(1,ids)/temperature,dim=1)
                    loss=loss+cfg.der_alpha*temperature**2*F.kl_div(student,past,reduction='batchmean')
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
                        anchor=client.ewc_anchor[name].to(device)
                        fisher=client.ewc_fisher[name].to(device)
                        pen=pen+(fisher*(p-anchor).square()).sum()
                loss=loss+.5*cfg.regularization_strength*pen
            loss.backward()
            if cfg.method in {'gem','agem'} and oldmemory:
                g=_flatten_grads(params)
                if cfg.method=='agem':
                    refidx=rng.choice(oldmemory,size=min(cfg.batch_size,len(oldmemory)),replace=False).tolist()
                    ref=_reference_gradient(model,params,bank.train,refidx,cfg.batch_size,device,
                                            cfg.seed+round_id+client.client_id)
                    z,last_projection,v=agem_project(g,ref)
                else:
                    refs=[]
                    # Each remembered class forms its own explicit constraint.
                    for cls in sorted(eligible_history):
                        mem=eligible_history[cls]
                        if mem:
                            refs.append(_reference_gradient(model,params,bank.train,mem,
                                                            cfg.batch_size,device,cfg.seed+cls))
                    z,last_projection,v=gem_project(g,refs)
                    qp_calls+=int(v>0)
                num_violations+=v
                _set_grads(params,z)
            optimizer.step()
            step_count+=1
            loss_total+=float(loss.detach().item())
    # Diagonal Fisher approximation for EWC, consolidated at each stage transition.
    if cfg.method=='ewc' and (round_id+1)%cfg.rounds_per_stage==0:
        if client.ewc_anchor is None or client.current_stage > client.ewc_stage:
            # Use a bounded local batch; this is an approximation, recorded in diagnostics.
            x,y=next(iter(_example_loader(bank.train,indices,min(cfg.batch_size,128),
                                          cfg.seed+round_id+client.client_id)))
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(device)),y.to(device)).backward()
            client.ewc_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            client.ewc_fisher={n:(p.grad.detach().cpu().square() if p.grad is not None else torch.zeros_like(p.cpu())) for n,p in model.named_parameters()}
            client.ewc_stage=client.current_stage
    if cfg.method=='mas' and (round_id+1)%cfg.rounds_per_stage==0:
        # MAS surrogate importance uses output sensitivity, independent of labels.
        xm,_=next(iter(_example_loader(bank.train,indices,min(cfg.batch_size,128),
                                      cfg.seed+round_id+client.client_id)))
        model.zero_grad(set_to_none=True)
        squared_output=model(xm.to(device)).square().sum(dim=1).mean()
        squared_output.backward()
        omega={n:(p.grad.detach().cpu().abs() if p.grad is not None else
                   torch.zeros_like(p.detach().cpu())) for n,p in model.named_parameters()}
        if client.mas_omega is not None:
            omega={n:omega[n]+client.mas_omega[n] for n in omega}
        client.mas_omega=omega
        client.mas_anchor={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    client.update_memory(indices,bank.train.targets,cfg.memory_per_class,rng,cfg.memory_policy)
    if cfg.method in {'der','derpp'} and client.memory:
        # Logits recorded at memory insertion/consolidation, never from test data.
        # This is a stage-consolidated local DER adaptation rather than the
        # paper's step-wise online reservoir update.
        retained=[int(i) for group in client.memory.values() for i in group]
        unseen=[i for i in retained if i not in client.memory_logits]
        if unseen:
            model.eval()
            with torch.no_grad():
                for batch_start in range(0,len(unseen),cfg.batch_size):
                    batch_ids=unseen[batch_start:batch_start+cfg.batch_size]
                    x=torch.stack([bank.train[i][0] for i in batch_ids]).to(device)
                    logits=model(x).detach().cpu()
                    for i,z in zip(batch_ids,logits):
                        client.memory_logits[int(i)]=z.clone()
            model.train()
    client.train_steps+=step_count
    return {'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'examples':len(indices),'steps':step_count,'loss':loss_total/max(1,step_count),
            'constraint_violations':num_violations,'qp_calls':qp_calls,
            'projection_norm_last':last_projection,'memory_samples':sum(map(len,client.memory.values())),
            'method_used':cfg.method,'fedprox_mu':cfg.fedprox_mu}
