"""LGA source radius calculation and accepted-only semantics."""
import pytest
import torch
from torch import nn
from cafcl.upstream_lga_radius import original_lga_radius_class,original_lga_radius


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,512))
        self.classifier=nn.Linear(512,100)
    def forward(self,x):return self.classifier(self.features(x))


class Bank:
    class Train:
        targets=[0]*50+[1]*50
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            from cafcl.models import NORMALIZATION_MEAN,NORMALIZATION_STD
            x=torch.full((3,32,32),(i+1)/105.)
            m=torch.tensor(NORMALIZATION_MEAN).view(3,1,1)
            s=torch.tensor(NORMALIZATION_STD).view(3,1,1)
            return (x-m)/s,self.targets[i]
    train=Train()


def test_author_lga_radius_source_functions_are_exact_ast():
    cls=original_lga_radius_class(cpu_test=True)
    assert cls.compute_radius.__code__.co_filename.endswith('LGA/myNetwork.py')
    assert cls.Image_transform.__code__.co_filename.endswith('LGA/myNetwork.py')


def test_original_lga_radius_rejects_unaccepted_data():
    with pytest.raises(RuntimeError,match='unaccepted class'):
        original_lga_radius(Tiny(),Bank(),list(range(50)),
                            torch.device('cpu'))


def test_original_lga_radius_runs_author_covariance_on_accepted_train_only():
    model=Tiny()
    sizes=[]
    hook=model.features.register_forward_pre_hook(
        lambda m,x:sizes.append(len(x[0])))
    try:
        radius=original_lga_radius(model,Bank(),range(100),
                                    torch.device('cpu'))
    finally:
        hook.remove()
    assert 0<=radius<1000
    assert sizes and set(sizes)=={64}
