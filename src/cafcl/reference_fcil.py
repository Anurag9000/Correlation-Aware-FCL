"""Independent synchronous, class-incremental federated reference.

Global tasks, label-Dirichlet training partition, sampled online clients,
private replay, reset local SGD, full seen-class competition. This is a
generic reference, not a claimed exact reproduction of a named paper.
"""
from collections import defaultdict
from dataclasses import asdict,dataclass
from pathlib import Path
import argparse
import hashlib
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from .identity import digest,dataset_identity,runtime_identity
from .models import create_model,copy_cpu_state
from .joint.checkpoint_io import durable_torch_save


@dataclass(frozen=True)
class ReferenceConfig:
    seed:int=0
    clients:int=10
    classes_per_task:int=10
    rounds_per_task:int=50
    epochs:int=5
    batch_size:int=64
    alpha:float=.5
    participation:float=.5
    lr:float=.03
    momentum:float=.9
    weight_decay:float=.0005
    memory_per_class:int=0
    competition:str='global_seen'
    dataset:str='cifar100'
    model:str='resnet18_cifar32'

    def validate(self,nclasses):
        if (self.clients<2 or self.classes_per_task<2 or nclasses%self.classes_per_task
                or min(self.rounds_per_task,self.epochs,self.batch_size)<1
                or not np.isfinite(self.alpha) or self.alpha<=0
                or not 0<self.participation<=1 or not self.lr>0
                or not 0<=self.momentum<1 or self.weight_decay<0 or self.memory_per_class<0
                or self.competition not in ('global_seen','local_seen','all_outputs')):
            raise ValueError('Invalid synchronous FCIL configuration')


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:15],16)


def compile_reference(targets,nclasses,cfg):
    cfg.validate(nclasses)
    labels=np.asarray(targets,dtype=np.int64)
    if set(labels.tolist())!=set(range(nclasses)):
        raise ValueError('Expected complete contiguous TRAIN labels')
    rng=np.random.default_rng(seed_for(cfg.seed,'partition'))
    stages=[];m=max(1,min(cfg.clients,round(cfg.clients*cfg.participation)))
    for t in range(nclasses//cfg.classes_per_task):
        classes=list(range(t*cfg.classes_per_task,(t+1)*cfg.classes_per_task))
        # Reject empty client assignments rather than silently change client
        # population or copy examples. Record conditioning on nonempty clients.
        for attempt in range(1000):
            parts=[[] for _ in range(cfg.clients)]
            for c in classes:
                ids=rng.permutation(np.flatnonzero(labels==c))
                counts=rng.multinomial(len(ids),rng.dirichlet(np.full(cfg.clients,cfg.alpha)))
                for k,chunk in enumerate(np.split(ids,np.cumsum(counts)[:-1])):
                    parts[k].extend(map(int,chunk))
            if all(parts):break
        else:raise ValueError('Dirichlet partition could not produce nonempty clients')
        cohorts=[sorted(random.Random(seed_for(cfg.seed,'cohort',t,r)).sample(range(cfg.clients),m))
                 for r in range(cfg.rounds_per_task)]
        stages.append({'stage':t,'classes':classes,'parts':[sorted(x) for x in parts],
                       'cohorts':cohorts,'nonempty_partition_attempts':attempt+1})
    ids=[i for s in stages for p in s['parts'] for i in p]
    if sorted(ids)!=list(range(len(targets))):raise RuntimeError('Partition lost or duplicated TRAIN samples')
    result={'protocol':'synchronous_dirichlet_fcil_v1','config':asdict(cfg),
            'targets_sha256':digest(list(map(int,targets))),'stages':stages}
    result['plan_hash']=digest(result)
    return result


def _avg(states,sizes):
    if not states or min(sizes)<=0:raise ValueError('Invalid FedAvg participants')
    total=sum(sizes)
    return {k:(sum((v[k].double()*n for v,n in zip(states,sizes)),torch.zeros_like(states[0][k],dtype=torch.float64))/total).to(states[0][k].dtype)
            if states[0][k].is_floating_point() else torch.stack([s[k] for s in states]).amax(0)
            for k in states[0]}


@torch.inference_mode()
def score(model,dataset,seen,batch_size,device):
    model.eval();classes=sorted(seen);ids=[i for i,c in enumerate(dataset.targets) if int(c) in seen]
    if not ids:raise ValueError('Empty evaluation panel')
    active=torch.tensor(classes,device=device)
    counts=defaultdict(int);correct=defaultdict(int)
    nll=0.;brier=0.;bins=[[0,0.,0.] for _ in range(15)]
    for x,y in DataLoader(Subset(dataset,ids),batch_size=batch_size,shuffle=False,
                          generator=torch.Generator().manual_seed(0)):
        y=y.to(device);logits=model(x.to(device))[:,active];prob=logits.softmax(1)
        true=(y[:,None]==active).long().argmax(1)
        conf,pred=prob.max(1);ok=pred.eq(true)
        nll+=float(F.cross_entropy(logits,true,reduction='sum'))
        brier+=float((prob-F.one_hot(true,len(classes))).square().sum())
        for c in y.unique().tolist():
            mask=y==c;counts[c]+=int(mask.sum());correct[c]+=int(ok[mask].sum())
        for k in range(15):
            mask=(conf*15).long().clamp(max=14)==k
            bins[k][0]+=int(mask.sum());bins[k][1]+=float(conf[mask].sum());bins[k][2]+=float(ok[mask].sum())
    n=len(ids)
    return {'accuracy':sum(correct.values())/n,'per_class':{c:correct[c]/counts[c] for c in classes if counts[c]},
            'nll':nll/n,'brier':brier/n,'ece_15':sum(abs(b[1]-b[2]) for b in bins)/n,
            'samples':n,'competitor_classes':classes}


def run_reference(cfg,bank,output,device='cpu'):
    from .simulator import atomic_json
    plan=compile_reference(bank.train.targets,bank.num_classes,cfg)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise RuntimeError('Use a fresh output root for this reference')
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'identity.json',{'runtime':runtime_identity(),'dataset':dataset_identity(bank),'plan_hash':plan['plan_hash']})
    atomic_json(out/'status.json',{'status':'running'})
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    model=create_model(cfg.model,bank.num_classes,allow_test_model=cfg.dataset=='synthetic',learning_setting='class_il').to(device)
    server=copy_cpu_state(model);seen=set();mem=[{} for _ in range(cfg.clients)]
    private_seen=[set() for _ in range(cfg.clients)];learned=[set() for _ in range(cfg.clients)]
    history=[];prior_panels={};steps=0
    for s in plan['stages']:
        t=s['stage'];seen.update(s['classes'])
        for r,cohort in enumerate(s['cohorts']):
            states=[];sizes=[]
            for cid in cohort:
                current=s['parts'][cid];learned[cid].update(current)
                private_seen[cid].update(int(bank.train.targets[i]) for i in current)
                replay=[i for group in mem[cid].values() for i in group]
                ids=current+replay
                model.load_state_dict(server);model.train()
                optimizer=torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
                competitors=seen if cfg.competition=='global_seen' else private_seen[cid] if cfg.competition=='local_seen' else set(range(bank.num_classes))
                active=torch.tensor(sorted(competitors),device=device)
                generator=torch.Generator().manual_seed(seed_for(cfg.seed,'batches',t,r,cid))
                loader=DataLoader(Subset(bank.train,ids),batch_size=cfg.batch_size,shuffle=True,generator=generator)
                for _ in range(cfg.epochs):
                    for x,y in loader:
                        x=x.to(device);y=y.to(device)
                        match=y[:,None]==active
                        if not bool(match.any(1).all()):raise RuntimeError('Training target missing from competitor set')
                        optimizer.zero_grad(set_to_none=True)
                        loss=F.cross_entropy(model(x)[:,active],match.long().argmax(1))
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite local objective')
                        loss.backward();optimizer.step();steps+=1
                states.append(copy_cpu_state(model));sizes.append(len(current))
            server=_avg(states,sizes);model.load_state_dict(server)
            history.append({'stage':t,'round':r,'cohort':cohort,'optimizer_steps':steps,
                            'validation':score(model,bank.validation,seen,cfg.batch_size,device)})
            atomic_json(out/'metrics.json',history)
        # Only examples actually optimized by an online client can enter its
        # private exemplar reservoir; offline samples are never replayed.
        for cid in range(cfg.clients):
            pool=defaultdict(list)
            for i in learned[cid]:pool[int(bank.train.targets[i])].append(i)
            mem[cid]={c:sorted(ids,key=lambda i:(seed_for(cfg.seed,'memory',cid,c,i),i))[:cfg.memory_per_class]
                      for c,ids in pool.items()}
        fixed={}
        for old,record in prior_panels.items():
            panel=score(model,bank.validation,set(record['competitor_classes']),cfg.batch_size,device)
            fixed[old]={'reference':record['accuracy'],'current':panel['accuracy'],'signed_drop':record['accuracy']-panel['accuracy']}
        history[-1]['fixed_context_retention']=fixed
        prior_panels[t]=history[-1]['validation']
        atomic_json(out/'metrics.json',history)
    model.load_state_dict(server)
    final=score(model,bank.test,seen,cfg.batch_size,device)
    durable_torch_save({'model_state':server,'config':asdict(cfg),'plan_hash':plan['plan_hash']},out/'model_final.pt')
    atomic_json(out/'summary.json',{'test':final,'validation':history[-1]['validation'],'optimizer_steps':steps,
        'fixed_context_retention':history[-1]['fixed_context_retention'],'plan_hash':plan['plan_hash']})
    atomic_json(out/'status.json',{'status':'completed'})
    return final


def main():
    from .data import get_data_bank
    from .simulator import atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','run']);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--device',default='cuda:0');p.add_argument('--download',action='store_true')
    a=p.parse_args();cfg=ReferenceConfig(**json.loads(Path(a.config).read_text()))
    bank=get_data_bank(cfg.dataset,a.data_root,download=a.download,seed=cfg.seed)
    if a.command=='plan':
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'plan.json',compile_reference(bank.train.targets,bank.num_classes,cfg))
    else:
        if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        run_reference(cfg,bank,a.output,a.device)


if __name__=='__main__':main()
