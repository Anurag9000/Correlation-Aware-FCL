"""Critical server CL-state synchronization invariants."""
import torch
from cafcl.federated_cl import FederatedCLState,add_importance
from cafcl.continual import LocalState

class Bank:
    class Train:
        targets=[0,1,0,1,2,3,2,3]
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):return torch.zeros(3,32,32),self.targets[i]
    train=Train()
    num_classes=4

def client(cid,mem=None):
    c=LocalState(cid)
    c.receive({'weight':torch.tensor([1.,2.])},0)
    c.private_memory=mem or {}
    c.pending_upload_classes=set(c.private_memory)
    return c

def test_replay_consolidates_raw_examples_without_echoes():
    a,b=client(0,{0:[0,2]}),client(1,{1:[1,3],2:[4]})
    server=FederatedCLState()
    server.merge_clients([a,b],Bank(),'replay',{'weight':torch.ones(2)},1)
    server.broadcast_to(a,'replay');server.broadcast_to(b,'replay')
    assert len(a.received_example_payloads)==5
    assert a.memory==b.memory
    assert server.statistics()['shared_replay_samples']==5
    server.merge_clients([a],Bank(),'replay',{'weight':torch.ones(2)},2)
    assert server.statistics()['shared_replay_samples']==5

def test_disconnected_client_waits_until_rejoin():
    a,b=client(0,{0:[0]}),client(1,{2:[4]})
    server=FederatedCLState()
    server.merge_clients([a],Bank(),'agem',{'weight':torch.ones(2)},1)
    server.broadcast_to(a,'agem')
    assert b.received_memory=={}
    server.merge_clients([b],Bank(),'agem',{'weight':torch.ones(2)},2)
    server.broadcast_to(b,'agem')
    assert set(b.received_gradient_supports)=={0,2}
    assert b.federated_cl_version==2

def test_fisher_is_added_once_and_anchor_is_precision_weighted():
    a,b=client(0),client(1)
    add_importance(a,'ewc',{'weight':torch.tensor([2.,4.])},
        {'weight':torch.tensor([1.,3.])})
    add_importance(b,'ewc',{'weight':torch.tensor([3.,2.])},
        {'weight':torch.tensor([4.,5.])})
    server=FederatedCLState()
    server.merge_clients([a,b],Bank(),'ewc',{'weight':torch.ones(2)},1)
    a.pending_importance={};b.pending_importance={}
    server.broadcast_to(a,'ewc')
    assert torch.equal(a.ewc_fisher['weight'],torch.tensor([5.,6.]))
    assert torch.allclose(a.ewc_anchor['weight'],torch.tensor([14/5,22/6]))
    server.merge_clients([a,b],Bank(),'ewc',{'weight':torch.ones(2)},2)
    assert torch.equal(server.importance['ewc']['precision']['weight'],torch.tensor([5.,6.]))

def test_der_and_lwf_share_history():
    a=client(0,{0:[0,2]})
    a.memory_logits={0:torch.arange(4.).float(),2:torch.arange(4.).float()}
    d=FederatedCLState()
    d.merge_clients([a],Bank(),'der',{'weight':torch.ones(2)},1)
    b=client(1)
    d.broadcast_to(b,'der')
    assert len(b.memory_logits)==2
    t=FederatedCLState()
    t.merge_clients([a],Bank(),'lwf',{'weight':torch.tensor([9.,8.])},1)
    t.broadcast_to(b,'lwf')
    assert torch.equal(b.lwf_teacher['weight'],torch.tensor([9.,8.]))
