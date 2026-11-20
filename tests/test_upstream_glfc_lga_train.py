"""Original GLFC/LGA local trainer really runs source SGD and loss."""
from types import SimpleNamespace
import torch
from torch import nn
import pytest

from cafcl.upstream_glfc_lga_memory import original_memory_methods
from cafcl.upstream_glfc_lga_train import train_original_glfc_lga
from cafcl.continual import LocalState


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,512),nn.Tanh())
        self.classifier=nn.Linear(512,100)
    def forward(self,x):return self.classifier(self.features(x))


class Bank:
    class Train:
        targets=[0]*3+[1]*3+[2]*3+[3]*3
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            return torch.full((3,8,8),float(i)/50),self.targets[i]
    train=Train()


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_authors_source_train_ast_and_optimizer_contents(method):
    source=original_memory_methods(method,cpu_test=True)
    assert source.train.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')
    assert source._compute_loss.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')
    assert 'SGD' in source.train.__code__.co_names
    assert 'step' in source.train.__code__.co_names


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_original_source_training_executes_and_resets_optimizer(method):
    model=SmallNet()
    state=LocalState(0)
    state.receive(model.state_dict(),0)
    state.arrival(list(range(6)),(0,1),0)
    state.local_experience=0
    cfg=SimpleNamespace(method=method,seed=0,epochs=1,lr=.03,
                        batch_size=64,memory_per_class=2,dataset='synthetic')
    output=train_original_glfc_lga(model,state,Bank(),cfg,torch.device('cpu'),
                                  0,None,experience_end=True,
                                  active_pair_classes=(0,1))
    assert output['steps']==1
    assert output['examples']==6
    assert output['optimizer_state_policy']=='authors_fresh_per_task'
    assert output['sgd_momentum']==0
    assert output['sgd_weight_decay']==.00001
    assert state.native_teacher is not None
    assert state.native_memory_state['last_classes']==[0,1]
    assert len(state.native_teacher)==len(model.state_dict())
    assert state.optimizer_state is None
