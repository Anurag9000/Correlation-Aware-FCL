"""Accepted-only original GLFC/LGA proxy server and teacher selection."""
from types import SimpleNamespace
import copy
import torch
from torch import nn
import pytest

from cafcl.upstream_native_proxy_server import (
    source_proxy_server_step,original_source_teacher_slots)
from cafcl.federated_cl import FederatedCLState
from cafcl.continual import LocalState
from cafcl.upstream_native_proxy import original_proxy_class


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*4*4,8))
        self.classifier=nn.Linear(8,100)
    def forward(self,x):
        return self.classifier(self.features(x))


def receipt(method,client,experience,cls):
    grad=torch.zeros((100,4))
    grad[cls]=-1
    return dict(method=method,client=client,local_experience=experience,
        cls=cls,gradients=[
            torch.zeros(1),grad,torch.zeros(100)])


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_two_accepted_receipts_defer_without_invoking_fake_proxy(method):
    server=FederatedCLState()
    for cls in (12,13):
        key=(0,1,cls)
        server.native_prototype_gradients[key]=receipt(method,0,1,cls)
    server.native_latest_accepted_prototypes=[(0,1,12),(0,1,13)]
    model=TinyModel()
    cfg=SimpleNamespace(seed=1,lr=.03)
    assert source_proxy_server_step(
        server,method,model,model.state_dict(),cfg,
        torch.device('cpu'),round_id=1) is None
    assert len(server.native_proxy_pending_receipts)==2
    assert server.native_proxy_state is None
    assert not server.native_proxy_history[-1]['ran_source_dataloader']
    assert server.native_proxy_history[-1]['reason'].startswith('original_monitor')
    # The offline client has two private receipts that never entered
    # native_prototype_gradients; they cannot be discovered here.
    hidden=LocalState(4)
    hidden.pending_native_prototypes=[receipt(method,4,1,38)]
    assert len(server.native_proxy_pending_receipts)==2


def test_source_proxy_teacher_slots_are_not_fabricated_when_unavailable():
    client=LocalState(0)
    client.native_teacher_classes={0,1}
    model=TinyModel()
    assert original_source_teacher_slots(client,model,torch.device('cpu'),
                                        (0,1,2,3),(2,3))==[None,None]


def test_proxy_teacher_slots_restore_distinct_authors_model_back_snapshots():
    client=LocalState(0)
    client.native_teacher_classes={0,1}
    model=TinyModel()
    a=copy.deepcopy(model.state_dict())
    b=copy.deepcopy(model.state_dict())
    a['classifier.bias']+=1
    b['classifier.bias']+=2
    client.received_native_proxy={
        'best_model_1':a,'best_model_2':b}
    result=original_source_teacher_slots(
        client,model,torch.device('cpu'),(0,1,2,3),(2,3))
    x=torch.randn(64,3,4,4)
    assert result[0](x).shape==(64,2)
    torch.testing.assert_close(result[1](x)-result[0](x),
                               torch.ones((64,2)),rtol=0,atol=1e-6)
    assert not any(p.requires_grad for m in result for p in m.parameters())


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_sufficient_accepted_receipts_call_actual_authors_dataloader(
    method,monkeypatch):
    server=FederatedCLState()
    keys=[]
    for cls in (12,13,14,15):
        key=(0,1,cls)
        keys.append(key)
        server.native_prototype_gradients[key]=receipt(method,0,1,cls)
    server.native_latest_accepted_prototypes=keys
    original=original_proxy_class(method,cpu_test=True)
    called=[]
    def fake_dataloader(self,pool,*args):
        called.append(len(pool))
        self.new_set=[[__import__('numpy').zeros(
            (32,32,3),dtype='uint8') for _ in range(20)] for _ in range(4)]
        self.new_set_label=[12,13,14,15]
        self.best_model_2=copy.deepcopy(self.model)
        self.best_perf=.75
    monkeypatch.setattr(original,'dataloader',fake_dataloader)
    model=TinyModel()
    cfg=SimpleNamespace(seed=1,lr=.03)
    result=source_proxy_server_step(
        server,method,model,model.state_dict(),cfg,
        torch.device('cpu'),round_id=1)
    assert called==[4]
    assert len(server.native_proxy_pending_receipts)==0
    assert result['source_provenance']=='authors_original_proxyServer.dataloader'
    assert result['reconstructed_images']==80
    assert result['best_model_2'] is not None
    assert result['server_round']==1
    assert server.native_proxy_history[-1]['source_reconstruction']
