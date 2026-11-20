"""FedCLASS original CE and exemplar constructor integration."""
import numpy as np
import pytest
import torch
from torch import nn

from cafcl.upstream_fedclass_ce import (
    original_class_method,update_original_exemplars,train_original_ce,
)


def test_upstream_train_function_is_original():
    fn=original_class_method('FedCLASS/models/ce.py','CE','_train')
    assert fn.__code__.co_filename.endswith('FedCLASS/models/ce.py')
    assert 'optimizer' in fn.__code__.co_names
    assert 'GradScaler' in fn.__code__.co_names


def test_upstream_exemplar_selector_exact_first_k_and_no_test_access():
    class Train:
        targets=[0,0,0,0,1,1,1,1]
        def __len__(self):return len(self.targets)
    class Bank:
        train=Train()
    from cafcl.continual import LocalState
    c=LocalState(client_id=0)
    c.private_memory={0:[0]}
    c.received_memory={3:[99]}
    update_original_exemplars(c,Bank(),list(range(4,8)),2)
    assert c.private_memory=={0:[0],1:[4,5]}
    assert c.memory=={0:[0],1:[4,5],3:[99]}


def test_upstream_first_k_is_not_reservoir_sampling():
    class Train:
        targets=[0]*8+[1]*8
        def __len__(self):return 16
    class Bank:
        train=Train()
    from cafcl.continual import LocalState
    c=LocalState(client_id=0)
    update_original_exemplars(c,Bank(),list(range(16)),3)
    assert c.private_memory=={0:[0,1,2],1:[8,9,10]}


def test_original_ce_direct_training_uses_known_class_competition_and_batch64():
    # Source CE._train resets its optimizer rather than carrying momentum
    # between tasks. A tiny CPU network is enough to validate the exact
    # training body; CUDA-only AMP automatically disables on CPU.
    from cafcl.continual import LocalState
    from cafcl.config import RunConfig
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*4*4,4),nn.ReLU())
            self.classifier=nn.Linear(4,100)
        def forward(self,x):return self.classifier(self.features(x))
    class Train:
        targets=[0]*5+[1]*5
        def __len__(self):return len(self.targets)
        def __getitem__(self,index):
            return torch.full((3,4,4),float(index)/10.),self.targets[index]
    class Bank:
        train=Train()
    model=Tiny()
    client=LocalState(client_id=0)
    client.receive(model.state_dict(),0)
    client.observed_classes={0,1}
    client.seen_indices=set(range(10))
    client.current_indices=list(range(10))
    cfg=RunConfig(dataset='synthetic',method='replay',batch_size=64,
                  epochs=1,lr=.03,momentum=.9,weight_decay=.0005,
                  memory_per_class=2,share_continual_state=True)
    result=train_original_ce(model,client,Bank(),cfg,torch.device('cpu'),
                             0,experience_end=True,active_pair_classes=(0,1))
    assert result['steps']==1
    assert result['examples']==10
    assert result['source_method']=='FedCLASS/models/ce.py:CE._train'
    assert client.private_memory=={0:[0,1],1:[5,6]}
    assert result['optimizer_state_policy']=='source_reset_each_task'
    assert all(torch.isfinite(v.float()).all() for v in result['weights'].values())
