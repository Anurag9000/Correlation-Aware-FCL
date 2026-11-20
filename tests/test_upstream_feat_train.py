"""Published FEAT local source SGD, geometry and explicit GSA shape fix."""
from types import SimpleNamespace
import torch
from torch import nn
import pytest

from cafcl.upstream_feat_train import (
    original_feat_train_methods,train_original_feat)
from cafcl.continual import LocalState


def test_source_train_bodies_provenance_and_source_optimizer():
    Source=original_feat_train_methods(cpu_test=True)
    for name in ('_local_update','_local_finetune'):
        m=getattr(Source,name)
        assert m.__code__.co_filename.endswith('FEAT/methods/FEAT.py')
        assert 'SGD' in m.__code__.co_names
        assert 'gsa_distill_loss' in m.__code__.co_names


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,12))
        self.classifier=nn.Linear(12,100)
    def forward(self,x):return self.classifier(self.features(x))


class Bank:
    class Train:
        targets=[0]*8+[1]*8+[2]*8+[3]*8
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            return torch.full((3,8,8),float(i)/40),self.targets[i]
    train=Train()


@pytest.mark.parametrize("stage",["first","second"])
def test_real_source_local_update_and_finetune_on_two_class_tasks(monkeypatch,stage):
    # The production selector requires official backbone attention features.
    # Patch only that independent selection mechanism to isolate the
    # original training source in CPU test fixtures.
    import cafcl.upstream_feat_replay as replay
    monkeypatch.setattr(replay,'source_feat_select_local',lambda *args,**kwargs:{})
    model=Tiny()
    state=LocalState(client_id=0)
    state.receive(model.state_dict(),0)
    cfg=SimpleNamespace(model='test_smallcnn',method='native_feat',
        dataset='synthetic',share_continual_state=True,
        epochs=1,lr=.03,weight_decay=.0005,batch_size=64,
        seed=0,memory_per_class=2)
    state.arrival(list(range(16)),(0,1),0)
    state.local_experience=0
    first=train_original_feat(model,state,Bank(),cfg,torch.device('cpu'),0,
        None,experience_end=True,active_pair_classes=(0,1))
    assert first['steps']==1
    assert first['source_method'].endswith('FEAT._local_update')
    assert first['sgd_momentum']==.9
    if stage=='second':
        state.shared_weights=first['weights']
        state.arrival(list(range(16,32)),(2,3),1)
        state.local_experience=1
        state.private_memory={0:[0,1],1:[8,9]}
        state.memory={0:[0,1],1:[8,9]}
        second=train_original_feat(model,state,Bank(),cfg,torch.device('cpu'),
             1,None,experience_end=True,active_pair_classes=(2,3))
        assert second['steps']==1
        assert second['source_method'].endswith('FEAT._local_finetune')
        assert second['source_gsa_shape_repair']
        assert state.pending_feat_ecc and state.pending_feat_ecc[-1]['PH'].shape==(12,12)
