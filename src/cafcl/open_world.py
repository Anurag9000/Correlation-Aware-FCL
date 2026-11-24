"""Open-world Task-IL reference with versioned, dynamically registered heads.

The coordinator knows the compiled schedule; learners receive only the task
event being executed, private replay, and actual server downloads. The server
discovers a private head only when an uploader contributes it. This is a
separate SGD/private-Replay reference, not a transparent migration of legacy
native methods or the historical raw-sharing simulator.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass,field
from pathlib import Path
import argparse
import json
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from .models import create_model,copy_cpu_state,task_head_key
from .reference_fcil import seed_for
from .identity import runtime_identity,dataset_identity,digest
from .joint.checkpoint_io import durable_torch_save
from .joint.task_evaluation import evaluate_task_conditioned
from .joint.adapter import load_plan


class OpenHeadBank(nn.Module):
    def __init__(self,width,num_classes,seed,mode='dynamic'):
        super().__init__()
        if mode not in ('dynamic','shared_masked'):raise ValueError('Unknown head mode')
        self.width=width;self.seed=seed;self.num_classes=num_classes;self.mode=mode
        self.task_classes={}
        self.heads=nn.ModuleDict()
        if mode=='shared_masked':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_for(seed,'shared_classifier'))
                self.shared=nn.Linear(width,num_classes)

    def register(self,head,classes):
        head=task_head_key(head);classes=tuple(sorted(set(map(int,classes))))
        if len(classes)<2 or min(classes)<0 or max(classes)>=self.num_classes:
            raise ValueError('Invalid task class registry')
        if head in self.task_classes:
            if self.task_classes[head]!=classes:raise ValueError('Conflicting task-head schema')
            return
        self.task_classes[head]=classes
        if self.mode=='dynamic':
            # Independent of arrival ordering, worker RNG and future task IDs.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_for(self.seed,'head',head,classes))
                self.heads[head]=nn.Linear(self.width,len(classes))

    def forward(self,features,head):
        if head not in self.task_classes:raise KeyError('Task head has not arrived')
        if self.mode=='dynamic':return self.heads[head](features)
        return self.shared(features)[:,list(self.task_classes[head])]


class OpenModel(nn.Module):
    def __init__(self,backbone,nclasses,seed,mode,registry=None,synthetic=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base=create_model(backbone,nclasses,allow_test_model=synthetic,learning_setting='class_il')
        self.features=base.features
        self.classifier=OpenHeadBank(self.features.num_features,nclasses,seed,mode)
        for h,cs in sorted((registry or {}).items()):self.classifier.register(h,cs)


@dataclass
class PrivateState:
    registry:dict=field(default_factory=dict)
    weights:dict=field(default_factory=dict)
    anchor:dict=field(default_factory=dict)
    version:int=0
    momentum:dict=field(default_factory=dict)
    memory:list=field(default_factory=list)
    pending_examples:int=0
    pending_steps:int=0
    pending_heads:dict=field(default_factory=dict)

    def download(self,registry,weights,version,reset=False):
        self.registry=dict(registry)
        self.weights={k:v.clone() for k,v in weights.items()}
        self.anchor={k:v.clone() for k,v in weights.items()}
        self.version=version;self.pending_examples=0;self.pending_steps=0;self.pending_heads={}
        self.momentum={} if reset else {k:v for k,v in self.momentum.items() if k in weights}
        if any(head not in self.registry for _,head in self.memory):
            raise RuntimeError('Download removed a previously trained private head')


def register_private(client,head,classes,factory):
    """Expand only this learner. Preserve parameters and SGD slots by name."""
    registry=dict(client.registry)
    if head in registry:
        if tuple(registry[head])!=tuple(classes):raise ValueError('Conflicting private task metadata')
        return
    registry[head]=tuple(classes)
    initialized=copy_cpu_state(factory(registry))
    if set(client.weights)-set(initialized):raise RuntimeError('Private schema contraction')
    for k,v in initialized.items():
        if k not in client.weights:
            client.weights[k]=v.clone();client.anchor[k]=v.clone()
    client.registry=registry


def merge_dynamic(server,registry,clients,version,factory,mode='dynamic',stale_policy='damped_delta'):
    """Contributor-only head aggregation; server head schema grows on upload."""
    if stale_policy not in ('damped_delta','undamped_delta','direct_model'):raise ValueError('Unknown stale policy')
    uploads=[c for c in clients if c.pending_steps>0]
    if not uploads:return dict(registry),{k:v.clone() for k,v in server.items()},version
    expanded=dict(registry)
    for c in uploads:
        if c.version>version:raise RuntimeError('Future client version')
        for head,n in c.pending_heads.items():
            if n<=0:continue
            classes=tuple(c.registry[head])
            if head in expanded and tuple(expanded[head])!=classes:raise ValueError('Conflicting uploaded head schema')
            expanded[head]=classes
    initial=copy_cpu_state(factory(expanded))
    base={k:(server[k] if k in server else initial[k]) for k in initial}
    output={}
    for name,g in base.items():
        is_head=mode=='dynamic' and name.startswith('classifier.heads.')
        head=name.split('.')[2] if is_head else None
        candidates=[];weights=[]
        for c in uploads:
            exposure=float(c.pending_heads.get(head,0)) if is_head else float(c.pending_examples)
            if exposure<=0:continue
            if name not in c.weights or name not in c.anchor:
                raise RuntimeError('Contributor missing parameter or anchor: '+name)
            local=c.weights[name];anchor=c.anchor[name];lag=version-c.version
            if local.shape!=g.shape or anchor.shape!=g.shape:raise RuntimeError('Dynamic parameter shape mismatch')
            if lag==0:
                if not torch.equal(anchor,g):raise RuntimeError('Fresh dynamic anchor differs from server')
                candidate=local
            elif stale_policy=='direct_model':candidate=local
            elif not g.is_floating_point() or name.endswith(('running_mean','running_var','num_batches_tracked')):
                candidate=g
            else:
                scale=1/(1+lag) if stale_policy=='damped_delta' else 1.
                candidate=(g.double()+scale*(local.double()-anchor.double())).to(g.dtype)
            candidates.append(candidate);weights.append(exposure)
        if not candidates:output[name]=g.clone()
        elif g.is_floating_point():
            output[name]=(sum((v.double()*n for v,n in zip(candidates,weights)),torch.zeros_like(g,dtype=torch.float64))/sum(weights)).to(g.dtype)
        else:output[name]=torch.stack(candidates).amax(0)
        if output[name].is_floating_point() and not torch.isfinite(output[name]).all():
            raise RuntimeError('Nonfinite dynamic aggregate')
    return expanded,output,version+1


def _train_event(client,event,factory,bank,device,options):
    head=event['head_id'];classes=tuple(event['classes']);ids=list(map(int,event['train_indices']))
    if not ids:raise ValueError('Empty dynamic local event')
    register_private(client,head,classes,factory)
    model=factory(client.registry).to(device);model.load_state_dict(client.weights);model.train()
    opt=torch.optim.SGD(model.parameters(),lr=options['lr'],momentum=options['momentum'],weight_decay=options['weight_decay'])
    if options['optimizer_policy']=='persist':
        for name,param in model.named_parameters():
            if name in client.momentum:opt.state[param]['momentum_buffer']=client.momentum[name].to(device).clone()
    rng=random.Random(seed_for(options['seed'],'event',event['occurrence_id']))
    fresh=[(i,head) for i in ids]
    for _ in range(options['epochs']):
        rows=list(fresh);rng.shuffle(rows)
        for offset in range(0,len(rows),options['batch_size']):
            current=rows[offset:offset+options['batch_size']]
            replay=[rng.choice(client.memory) for _ in range(len(current))] if client.memory else []
            examples=current+replay
            x=torch.stack([bank.train[i][0] for i,h in examples]).to(device)
            y=torch.tensor([int(bank.train.targets[i]) for i,h in examples],device=device)
            features=model.features(x);loss=torch.zeros((),device=device)
            by_head=defaultdict(list)
            for j,(idx,h) in enumerate(examples):by_head[h].append(j)
            for h,positions in sorted(by_head.items()):
                indices=torch.tensor(positions,device=device);active=torch.tensor(client.registry[h],device=device)
                matches=y[indices,None]==active[None,:]
                if not bool(matches.any(1).all()):raise ValueError('Replay task/label provenance mismatch')
                logits=model.classifier(features[indices],h)
                loss=loss+F.cross_entropy(logits,matches.long().argmax(1),reduction='sum')/len(examples)
                client.pending_heads[h]=client.pending_heads.get(h,0)+len(positions)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite open-world loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            client.pending_steps+=1
    client.weights=copy_cpu_state(model)
    client.momentum={name:opt.state[p]['momentum_buffer'].detach().cpu().clone()
                     for name,p in model.named_parameters() if 'momentum_buffer' in opt.state[p]}
    client.pending_examples+=len(ids)
    # Bound by raw image count per class, then keep all legitimate source
    # head associations for retained images. No server raw-data exchange.
    by_class=defaultdict(set)
    for idx,h in client.memory+fresh:by_class[int(bank.train.targets[idx])].add((idx,h))
    memory=[]
    for cls,items in sorted(by_class.items()):
        unique=sorted({i for i,h in items},key=lambda i:(seed_for(options['seed'],'memory',cls,i),i))
        retained=set(unique[:options['memory_per_class']])
        memory.extend(sorted((i,h) for i,h in items if i in retained))
    client.memory=memory


def run_open_world(plan,bank,output,device='cpu',*,backbone='resnet18_cifar32',mode='dynamic',
                   epochs=1,batch_size=64,lr=.03,momentum=.9,weight_decay=.0005,
                   memory_per_class=0,optimizer_policy='persist',stale_policy='damped_delta',synthetic=False):
    from .simulator import atomic_json
    if plan['allocation_audit'].get('control_kind','federated')!='federated' or plan['allocation_audit'].get('epoch_slicing'):
        raise ValueError('Open-world reference supports ordinary federated atomic tasks only')
    if min(epochs,batch_size)<1 or memory_per_class<0 or not math.isfinite(lr) or lr<=0 or not 0<=momentum<1 or weight_decay<0:
        raise ValueError('Invalid open-world optimizer configuration')
    if optimizer_policy not in ('persist','reset_on_sync','reset_on_task'):raise ValueError('Unknown optimizer policy')
    seed=plan['seed'];options=dict(seed=seed,epochs=epochs,batch_size=batch_size,lr=lr,
        momentum=momentum,weight_decay=weight_decay,memory_per_class=memory_per_class,
        optimizer_policy=optimizer_policy)
    def factory(registry):return OpenModel(backbone,bank.num_classes,seed,mode,registry,synthetic)
    server=copy_cpu_state(factory({}));registry={};version=0
    clients=[PrivateState() for _ in range(plan['num_clients'])]
    for c in clients:c.download(registry,server,version)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise RuntimeError('Use fresh output root')
    atomic_json(out/'identity.json',{'runtime':runtime_identity(),'dataset':dataset_identity(bank),'plan_hash':plan['plan_hash'],
        'options':options,'mode':mode,'backbone':backbone,'stale_policy':stale_policy})
    atomic_json(out/'status.json',{'status':'running'})
    lookup={x['occurrence_id']:x for x in plan['tasks']};history=[];peaks={};arrived=set()
    for interval in plan['intervals']:
        for oid in interval['occurrences']:
            event=lookup[oid];client=clients[event['client_id']]
            if optimizer_policy=='reset_on_task':client.momentum={}
            _train_event(client,event,factory,bank,device,options)
            arrived.add(event['head_id'])
        cohort=interval['upload_clients']
        uploaders=[clients[c] for c in cohort if clients[c].pending_steps>0]
        registry,server,version=merge_dynamic(server,registry,uploaders,version,factory,mode,stale_policy)
        for cid in cohort:clients[cid].download(registry,server,version,reset=optimizer_policy=='reset_on_sync')
        model=factory(registry).to(device);model.load_state_dict(server)
        validation=evaluate_task_conditioned(model,bank,registry,batch_size,device)
        drops=[max(0.,peaks[h]-x['accuracy']) for h,x in validation['per_head'].items()
               if h in peaks and x['accuracy'] is not None]
        for h,x in validation['per_head'].items():
            if x['accuracy'] is not None:peaks[h]=max(peaks.get(h,0.),x['accuracy'])
        history.append({'interval':interval['interval_id'],'version':version,'head_registry':registry,
            'arrived_head_count':len(arrived),'integrated_head_count':len(registry),
            'head_coverage':len(registry)/len(arrived) if arrived else None,
            'classifier_parameters':sum(p.numel() for p in model.classifier.parameters()),
            'validation':validation,'fixed_head_forgetting':sum(drops)/len(drops) if drops else None})
        atomic_json(out/'metrics.json',history)
        durable_torch_save({'server':server,'registry':registry,'version':version,'clients':clients,
            'interval':interval['interval_id'],'history':history,'peaks':peaks,'plan_hash':plan['plan_hash']},out/'checkpoint_latest.pt')
    final=evaluate_task_conditioned(model,bank,registry,batch_size,device,'test')
    atomic_json(out/'summary.json',{'test_integrated_heads':final,'arrived_head_count':len(arrived),
        'integrated_head_count':len(registry),'head_coverage':len(registry)/len(arrived) if arrived else None,
        'model_mode':mode,'evaluation_protocol':'task_conditioned_integrated_heads_only'})
    durable_torch_save({'model_state':server,'registry':registry,'mode':mode,'plan_hash':plan['plan_hash']},out/'model_final.pt')
    atomic_json(out/'status.json',{'status':'completed'})
    return final


def main():
    from .data import get_data_bank
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='data');p.add_argument('--device',default='cuda:0')
    p.add_argument('--mode',choices=['dynamic','shared_masked'],default='dynamic')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--memory-per-class',type=int,default=0)
    p.add_argument('--optimizer-policy',choices=['persist','reset_on_sync','reset_on_task'],default='persist')
    p.add_argument('--stale-policy',choices=['damped_delta','undamped_delta','direct_model'],default='damped_delta')
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    raw=json.loads(Path(a.plan).read_text());bank=get_data_bank('cifar100',a.data_root,download=a.download,seed=raw['seed'])
    plan=load_plan(a.plan,raw['plan_hash'],bank.train.targets)
    run_open_world(plan,bank,a.output,a.device,mode=a.mode,epochs=a.epochs,batch_size=a.batch_size,
                   memory_per_class=a.memory_per_class,optimizer_policy=a.optimizer_policy,stale_policy=a.stale_policy)


if __name__=='__main__':main()
