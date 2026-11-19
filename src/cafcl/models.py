"""Single, paper-standard DeiT-Tiny/16 research backbone for every CIFAR-100 job.

DeiT: Touvron et al., Training data-efficient image transformers &
 distillation through attention, ICML 2021. ImageNet-1K pretrained timm model.
Non-ViT toy models below are strictly for synthetic regression tests, never
valid production CIFAR-100 configurations.
"""
from __future__ import annotations
import hashlib
import os
from torch import nn
import torch

BACKBONE_ID = 'deit_tiny_patch16_224.fb_in1k'
INPUT_SIZE = 224
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def _state_sha256(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0')
        digest.update(str(tuple(value.shape)).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PaperDeiTTiny(nn.Module):
    """Published DeiT-Tiny backbone + a single canonical CIFAR-100 classifier.

    Loading is fail-closed: no silently randomized or CNN replacement for
    missing pretrained weights, a missing timm installation, or bad dimensions.
    """
    def __init__(self, classes: int, *, pretrained: bool=True):
        super().__init__()
        if not pretrained:
            raise ValueError('Production DeiT backbone requires pretrained ImageNet-1K initialization')
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError('timm is required: pip install -e ".[test]"') from exc
        try:
            self.features=timm.create_model(BACKBONE_ID,pretrained=True,num_classes=0)
        except Exception as exc:
            raise RuntimeError('Pretrained DeiT-Tiny download/cache failed; refusing uninitialized fallback') from exc
        dim=getattr(self.features,'num_features',None)
        if dim != 192:
            raise RuntimeError(f'Expected DeiT-Tiny feature dimension 192, got {dim!r}')
        self.pretrained_backbone_sha256=_state_sha256(self.features.state_dict())
        expected=os.environ.get('CAFCL_EXPECTED_DEIT_SHA256','').strip()
        if expected and expected!=self.pretrained_backbone_sha256:
            raise RuntimeError('Pretrained backbone SHA-256 differs from the frozen study reference')
        self.classifier=nn.Linear(dim,classes)
        self.backbone_description=dict(model_id=BACKBONE_ID,pretrained=True,
            sha256=self.pretrained_backbone_sha256,input_size=INPUT_SIZE,
            normalization_mean=NORMALIZATION_MEAN,normalization_std=NORMALIZATION_STD,
            timm_version=timm.__version__)

    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,INPUT_SIZE,INPUT_SIZE):
            raise ValueError('DeiT expects (N,3,224,224) normalized images')
        return self.classifier(self.features(x))


def create_model(name: str, classes: int, *, allow_test_model: bool=False):
    if name==BACKBONE_ID:
        return PaperDeiTTiny(classes)
    if not allow_test_model:
        raise ValueError(f'Only {BACKBONE_ID} is allowed in CIFAR research jobs (got {name})')
    # Cheap, named test doubles for synthetic unit tests; never experiment jobs.
    if name=='test_linear':
        return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,classes))
    if name=='test_mlp':
        return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,256),nn.ReLU(),nn.Linear(256,classes))
    if name=='test_smallcnn':
        return nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(8,classes))
    raise ValueError(name)


def copy_cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def check_finite(state):
    return all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
