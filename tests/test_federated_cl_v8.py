"""v8: post-sync *first* experience, heterogeneous artifact fusion, and SI consolidation."""
from __future__ import annotations
from types import SimpleNamespace
import torch
import pytest
from cafcl.config import RunConfig
from cafcl.continual import LocalState,local_train
from cafcl.federated_cl import FederatedCLState,add_importance
from cafcl.models import create_model

class TinyBank:
    num_classes=4
    class Train:
        targets=[0,0,1,1,2,2,3,3]
        def __len__(self):return 8
        def __getitem__(self,i):
            # Different class-associated patterns, no test data.
            x=torch.zeros(3,32,32)
            x[self.targets[i]%3,self.targets[i]:self.targets[i]+3,:]=.5
            return x,self.targets[i]
    train=Train()

def model_state():
    torch.manual_seed(1701)
    return create_model('test_mlp',4,allow_test_model=True).state_dict()

def synced_client(cid=1):
    model=create_model('test_mlp',4,allow_test_model=True)
    model.load_state_dict(model_state())
    client=LocalState(cid)
    client.receive(model.state_dict(),0)
    return model,client

def cfg(method,**kwargs):
    return RunConfig(dataset='synthetic',model='test_mlp',num_clients=2,
        method=method,aggregator='fedavg',batch_size=1024,micro_batch_size=4,
        epochs=1,lr=.01,memory_per_class=2,share_continual_state=True,
        class_pair_protocol='strict',**kwargs)

def train_first_after_sync(model,client,method,**kwargs):
    client.local_experience+=1
    client.arrival([4,5,6,7],[2,3],stage=1)
    client.sample_first_local_experience.update({i:client.local_experience for i in [4,5,6,7]})
    return local_train(model,client,TinyBank(),cfg(method,**kwargs),torch.device('cpu'),1,
        experience_end=True,active_pair_classes=(2,3))

def test_all_memory_methods_start_post_sync_with_global_history(monkeypatch):
    import cafcl.continual as cl
    gradient_calls=[]
    original=cl._reference_gradient
    def traced_reference(*a,**kw):
        gradient_calls.append(tuple(a[3]))
        return original(*a,**kw)
    monkeypatch.setattr(cl,'_reference_gradient',traced_reference)
    for method in ('replay','agem','gem','er_ace','der','derpp'):
        gradient_calls.clear()
        model,client=synced_client()
        sender=LocalState(0)
        sender.receive(model.state_dict(),0)
        sender.private_memory={0:[0,1],1:[2,3]}
        sender.sample_first_local_experience={i:0 for i in range(4)}
        sender.memory_logits={i:torch.zeros(4) for i in range(4)}
        sender.last_learning_method=method
        sender.pending_upload_classes={0,1}
        server=FederatedCLState()
        server.merge_clients([sender],TinyBank(),method,model.state_dict(),1)
        server.broadcast_to(client,method)
        assert set(client.received_memory)=={0,1}
        assert set(client.trained_classes)=={0,1}
        assert client.received_sample_origins[0]==(0,0)
        assert sum(map(len,client.memory.values()))==4
        assert len(client.received_example_payloads)==4
        assert client.federated_cl_version==1
        if method in {'gem','agem'}:
            assert set(client.received_gradient_supports)=={0,1}
        if method in {'der','derpp'}:
            assert len(client.memory_logits)==4
        result=train_first_after_sync(model,client,method,**({'gem_grouping':'stage'} if method=='gem' else {}))
        assert result['steps']==1
        assert result['memory_samples']>=4
        if method in {'gem','agem'}:
            assert gradient_calls, f'{method} did not consult remotely synced constraints'
            assert any(set(k).issubset(set(range(4))) for k in gradient_calls)


def test_si_consolidation_is_sent_immediately_at_sync():
    model,client=synced_client()
    client.local_experience=0
    client.arrival([0,1,2,3],[0,1],0)
    result=local_train(model,client,TinyBank(),cfg('si'),torch.device('cpu'),0,
        active_pair_classes=(0,1),experience_end=True)
    assert result['steps']==1
    assert 'si' in client.pending_importance
    fresh=client.pending_importance['si']['precision']
    assert any(v.abs().sum()>0 for v in fresh.values())
    server=FederatedCLState()
    client.pending_upload_classes={0,1}
    server.merge_clients([client],TinyBank(),'si',result['weights'],1)
    server.broadcast_to(client,'si')
    assert 'si' in client.received_importance
    assert any(v.abs().sum()>0 for v in server.importance['si']['precision'].values())


def test_mixed_method_server_union_has_all_artifacts():
    bank=TinyBank()
    model,a=synced_client(0)
    _,b=synced_client(1)
    _,c=synced_client(2)
    _,d=synced_client(3)
    a.last_learning_method='replay'
    a.private_memory={0:[0],1:[2]}
    a.pending_upload_classes={0,1}
    b.last_learning_method='ewc'
    add_importance(b,'ewc',{'3.bias':torch.tensor([2.,1.,2.,1.])},
        {'3.bias':torch.tensor([3.,4.,3.,4.])})
    c.last_learning_method='lwf'
    d.last_learning_method='agem'
    d.private_memory={2:[4],3:[6]}
    d.pending_upload_classes={2,3}
    server=FederatedCLState()
    server.merge_clients([a,b,c,d],bank,'sgd',model.state_dict(),1)
    assert set(server.memory)=={0,1,2,3}
    assert 'ewc' in server.importance
    assert server.teacher_version==1
    # Everyone receives the complete passive snapshot, while only a matching
    # learner activates the corresponding regularizer/replay algorithm.
    for local,method in [(a,'replay'),(b,'ewc'),(c,'lwf'),(d,'agem')]:
        server.broadcast_to(local,method)
        assert set(local.received_memory)=={0,1,2,3}
        assert set(local.received_importance)=={'ewc'}
        assert local.received_lwf_teacher_version==1
        assert set(local.received_gradient_supports)=={0,1,2,3}
    assert set(b.ewc_fisher)=={'3.bias'}
    assert c.lwf_teacher_from_server


def test_disconnected_retains_private_later_receives_all():
    model,a=synced_client(0)
    _,b=synced_client(1)
    a.private_memory={0:[0]}
    b.private_memory={2:[4]}
    a.last_learning_method=b.last_learning_method='agem'
    server=FederatedCLState()
    server.merge_clients([a],TinyBank(),'agem',model.state_dict(),1)
    server.broadcast_to(a,'agem')
    assert b.federated_cl_version==0 and not b.received_memory
    assert b.private_memory=={2:[4]}
    b.private_memory[3]=[6]
    server.merge_clients([b],TinyBank(),'agem',model.state_dict(),2)
    server.broadcast_to(b,'agem')
    assert set(b.memory)=={0,2,3}
    assert b.private_memory=={2:[4],3:[6]}
    assert b.federated_cl_version==2


def test_server_echo_does_not_double_importance_or_archive():
    model,a=synced_client(0)
    a.last_learning_method='ewc'
    add_importance(a,'ewc',{'3.bias':torch.tensor([2.,2.,2.,2.])},{'3.bias':torch.tensor([3.,3.,3.,3.])})
    server=FederatedCLState()
    server.merge_clients([a],TinyBank(),'ewc',model.state_dict(),1)
    a.receive(model.state_dict(),1)
    server.broadcast_to(a,'ewc')
    server.merge_clients([a],TinyBank(),'ewc',model.state_dict(),2)
    assert torch.allclose(server.importance['ewc']['precision']['3.bias'],torch.full((4,),2.))

@pytest.mark.parametrize('method',('ewc','si','mas','lwf'))
def test_first_after_broadcast_uses_remote_importance_or_teacher(method,monkeypatch):
    model,local=synced_client(1)
    sender=LocalState(0)
    sender.receive(model.state_dict(),0)
    sender.last_learning_method=method
    server=FederatedCLState()
    server.integrated_classes={0,1}
    if method in ('ewc','si','mas'):
        key='3.bias'
        add_importance(sender,method,{key:torch.ones(4)},
            {key:torch.ones(4)*3})
    server.merge_clients([sender],TinyBank(),method,model.state_dict(),1)
    # Simulate full model and state sync together.
    local.receive(model.state_dict(),1)
    server.broadcast_to(local,method)
    assert {0,1}.issubset(local.trained_classes)
    if method=='lwf':
        assert local.lwf_teacher_from_server
        assert local.received_lwf_teacher_version==1
    else:
        assert torch.allclose(getattr(local,'ewc_fisher' if method=='ewc' else method+'_omega')['3.bias'],torch.ones(4))
    result=train_first_after_sync(model,local,method)
    assert result['steps']==1
    if method=='lwf':
        assert not local.lwf_teacher_from_server
        assert local.lwf_teacher is not None
