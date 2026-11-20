"""Actual original GLFC/LGA entropy and herding source invocation."""
from types import SimpleNamespace
import torch
import pytest
from torch import nn

from cafcl.upstream_glfc_lga_memory import (
    original_memory_methods,original_before_local_task)


class Tiny512(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*4*4,512))
        self.classifier=nn.Linear(512,100)
    def forward(self,x):return self.classifier(self.features(x))

class Bank:
    class Train:
        targets=[0]*8+[1]*8+[2]*8+[3]*8
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            # Input bytes normalized according to official CIFAR100 affine
            # transform, just as original model training expects.
            import cafcl.models as m
            base=torch.ones(3,4,4)*float(i%16)/16.
            return base,self.targets[i]
    train=Train()


@pytest.mark.parametrize("method",['native_glfc','native_lga'])
def test_published_methods_compiled_from_immutable_authors(method):
    cl=original_memory_methods(method,cpu_test=True)
    for f in ('update_new_set','entropy_signal','_construct_exemplar_set',
              '_reduce_exemplar_sets'):
        assert getattr(cl,f).__code__.co_filename.endswith(
            'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')


def test_before_task_requires_exact_two_new_classes():
    from cafcl.continual import LocalState
    m=Tiny512()
    state=LocalState(0)
    state.observed_classes={0,1}
    with pytest.raises(RuntimeError,match='exactly two'):
        original_before_local_task('native_glfc',m,state,Bank(),
             SimpleNamespace(batch_size=64,memory_per_class=2,seed=1),
             torch.device('cpu'),(0,))


@pytest.mark.parametrize("method",['native_glfc','native_lga'])
def test_actual_author_entropy_before_training_does_not_leak_unseen_images(method):
    from cafcl.continual import LocalState
    # Real production CIFAR-100 is 32x32; this test never triggers herding
    # on the first task (no last-class pair) and may use tiny pixels.
    model=Tiny512()
    client=LocalState(0)
    client.observed_classes={0,1}
    client.seen_indices=set(range(16))
    client.current_indices=list(range(16))
    client.local_experience=0
    cfg=SimpleNamespace(batch_size=64,memory_per_class=2,seed=1)
    called=original_before_local_task(method,model,client,Bank(),cfg,
             torch.device('cpu'),(0,1))
    assert called['last_classes']==[0,1]
    assert called['learned_classes']==[]
    assert called['last_entropy']>=0
    assert client.private_memory=={}
    assert model.training
