"""Production ViT contract, pinned initialization and exact logical batch semantics.

The stub isolates wrapper/integration tests from pretrained-weight downloads.
It is not a claim to have executed a real pretrained DeiT forward pass.
"""
from dataclasses import replace
import sys
import types
import pytest
import torch
from torch import nn
from cafcl.config import RunConfig
from cafcl.models import BACKBONE_ID, PaperDeiTTiny, create_model
from cafcl.full_matrix import expand
from cafcl.class_indexed import classifier_parameter_names,canonical_active_loss
from cafcl.continual import LocalState,local_train
from cafcl.data import get_data_bank

class FakeTimmFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features=192
        self.proj=nn.Linear(3,192)
    def forward(self,x):
        return self.proj(x.mean(dim=(2,3)))


def test_all_cifar_research_jobs_use_single_published_vit():
    jobs=list(expand(seeds=(0,),families=('balanced','task_length','offline_duration','local_optimization')))
    assert len(jobs)>5000
    assert {cfg.model for _,cfg in jobs}=={BACKBONE_ID}
    assert {cfg.micro_batch_size for _,cfg in jobs}=={16}
    assert {cfg.batch_size for _,cfg in jobs}=={512}
    assert all(cfg.stream_protocol for _,cfg in jobs)


@pytest.mark.parametrize('old_model',['smallcnn','linear','mlp','test_linear','test_mlp','test_smallcnn'])
def test_cifar_production_forbids_non_deit(old_model):
    with pytest.raises(AssertionError,match='DeiT'):
        replace(RunConfig(),model=old_model).validate()
    with pytest.raises(ValueError,match='Only'):
        create_model(old_model,100)


def test_model_requires_pretrained_weights():
    with pytest.raises(ValueError,match='pretrained'):
        PaperDeiTTiny(100,pretrained=False)


def test_deit_wrapper_uses_pretrained_timm_and_canonical_rows(monkeypatch):
    calls=[]
    def fake_create_model(name,pretrained,num_classes):
        calls.append((name,pretrained,num_classes))
        return FakeTimmFeatures()
    monkeypatch.setitem(sys.modules,'timm',types.SimpleNamespace(create_model=fake_create_model,__version__='1.0.31'))
    torch.manual_seed(0)
    model=create_model(BACKBONE_ID,100)
    assert calls==[(BACKBONE_ID,True,0)]
    assert model.backbone_description['model_id']==BACKBONE_ID
    assert len(model.pretrained_backbone_sha256)==64
    assert classifier_parameter_names(model)==('classifier.weight','classifier.bias')
    x=torch.randn(2,3,224,224)
    y=model(x)
    assert y.shape==(2,100)
    loss=canonical_active_loss(y,torch.tensor([6,25]),(6,25))
    loss.backward()
    assert torch.count_nonzero(model.classifier.weight.grad[[0,1,99]])==0
    assert torch.count_nonzero(model.classifier.weight.grad[[6,25]])>0
    assert torch.count_nonzero(model.features.proj.weight.grad)>0
    with pytest.raises(ValueError,match='224'):
        model(torch.randn(1,3,32,32))


def test_deit_missing_download_never_falls_back_to_random_model(monkeypatch):
    def fail(*args,**kwargs):raise OSError('no pretrained weights cached')
    monkeypatch.setitem(sys.modules,'timm',types.SimpleNamespace(create_model=fail,__version__='1.0.31'))
    with pytest.raises(RuntimeError,match='refusing uninitialized fallback'):
        create_model(BACKBONE_ID,100)


def test_microbatch_matches_logical_sgd_gradient_on_synthetic():
    bank=get_data_bank('synthetic','unused')
    torch.manual_seed(12)
    initial=create_model('test_linear',10,allow_test_model=True).state_dict()
    trained=[]
    for micro in (2,20):
        cfg=RunConfig(dataset='synthetic',model='test_linear',batch_size=20,
                micro_batch_size=micro,method='sgd',epochs=1,seed=24,lr=.01)
        m=create_model('test_linear',10,allow_test_model=True)
        m.load_state_dict(initial)
        c=LocalState(0)
        c.receive(initial,0)
        c.arrival(list(range(20)),[0,1],0)
        result=local_train(m,c,bank,cfg,torch.device('cpu'),round_id=0)
        assert result is not None
        trained.append({k:v.clone() for k,v in m.state_dict().items()})
    for key in trained[0]:
        assert torch.allclose(trained[0][key],trained[1][key],atol=1e-6,rtol=1e-5)
