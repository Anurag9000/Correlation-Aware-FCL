import torch
from cafcl.open_world import OpenModel,PrivateState,register_private,merge_dynamic
from cafcl.models import copy_cpu_state

def factory(registry):
    return OpenModel('test_linear',10,7,'dynamic',registry,True)

def test_dynamic_stale_schema_and_momentum_names():
    initial=copy_cpu_state(factory({}));server=initial;registry={}
    a=PrivateState();b=PrivateState()
    for c in (a,b):c.download(registry,server,0)
    h1='task_000_001';h2='task_002_003'
    register_private(a,h1,(0,1),factory)
    register_private(b,h2,(2,3),factory)
    a.pending_steps=1;a.pending_examples=2;a.pending_heads[h1]=2
    b.pending_steps=1;b.pending_examples=2;b.pending_heads[h2]=2
    key=f'classifier.heads.{h2}.bias'
    old=b.anchor[key].clone()
    b.weights[key]=old+2
    registry,server,version=merge_dynamic(server,registry,[a],0,factory)
    assert h2 not in registry and key not in server
    registry,server,version=merge_dynamic(server,registry,[b],version,factory)
    assert set(registry)=={h1,h2} and version==2
    assert torch.allclose(server[key],old+1)
    b.momentum[key]=torch.ones_like(old)
    b.download(registry,server,version)
    assert torch.equal(b.momentum[key],torch.ones_like(old))
    assert b.pending_steps==0

def test_fixed_capacity_masked_head():
    a=OpenModel('test_linear',10,7,'shared_masked',{},True)
    n=sum(p.numel() for p in a.classifier.parameters())
    a.classifier.register('task_001_002',(1,2))
    a.classifier.register('task_003_004',(3,4))
    assert sum(p.numel() for p in a.classifier.parameters())==n
    assert a.classifier(torch.zeros(3,3072),'task_001_002').shape==(3,2)
