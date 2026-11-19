"""Reference gradients must not silently update CNN BatchNorm buffers."""
import torch
from torch import nn
from cafcl.continual import _reference_gradient
from cafcl.data import TinySynthetic


def test_agem_reference_gradient_preserves_running_statistics():
    model=nn.Sequential(
        nn.Conv2d(3,4,3,stride=2,padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2,2)),
        nn.Flatten(),
        nn.Linear(16,10))
    model.train()
    bn=model[1]
    initial=(bn.running_mean.clone(),bn.running_var.clone(),bn.num_batches_tracked.clone())
    grad=_reference_gradient(model,list(model.parameters()),TinySynthetic(True,0),
                             list(range(8)),8,torch.device('cpu'),23)
    assert torch.count_nonzero(grad)>0
    assert torch.equal(bn.running_mean,initial[0])
    assert torch.equal(bn.running_var,initial[1])
    assert torch.equal(bn.num_batches_tracked,initial[2])
