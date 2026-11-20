"""Execute official LGA/myNetwork.py:network.compute_radius, unchanged.

The paper's geometric radius is computed at the beginning of the second
local class stage from classes 0,1 after first server synchronization.
Only accepted TRAIN images are eligible; hidden clients' private samples
are prohibited before their upload. The source operates in 50-sample
analytical chunks; the compatible feature wrapper pads the actual neural
forward to 64, the user's physical-model-forward requirement.

Original mathematical covariance/trace/sqrt is unchanged. The number of
source chunks is 9 for the 450 TRAIN examples/class in the requested
90/10 train/validation experiment (original repo has 500 and uses 10).
This explicit data-protocol change is NOT bitwise original code behavior.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2]


class _CPUTransferOnly(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            node.func.attr='to'
            node.args=[ast.Constant(value='cpu')]
            node.keywords=[]
        return node


@lru_cache(maxsize=None)
def original_lga_radius_class(*,cpu_test=False):
    path=ROOT/'LGA/myNetwork.py'
    tree=ast.parse(path.read_text(),filename=str(path))
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='network']
    if len(cls)!=1:raise RuntimeError('Official LGA network missing')
    required={'Image_transform','compute_radius'}
    funcs=[n for n in cls[0].body
           if isinstance(n,ast.FunctionDef) and n.name in required]
    if {n.name for n in funcs}!=required:
        raise RuntimeError('Official LGA geometric methods not found')
    mod=ast.Module(body=[ast.ClassDef(name='OriginalGeometry',bases=[],
                   keywords=[],body=funcs,decorator_list=[])],type_ignores=[])
    if cpu_test:mod=_CPUTransferOnly().visit(mod)
    ast.fix_missing_locations(mod)
    env={'torch':torch,'nn':nn,'F':F,'np':np,'Image':Image}
    exec(compile(mod,str(path),'exec'),env)
    return env['OriginalGeometry']


class _Strict64FeatureNetwork(nn.Module):
    """Adapt author's analytical 50-example feature calls to model batch64."""
    def __init__(self,model):
        super().__init__()
        self.model=model
    def feature_extractor(self,x):
        n=len(x)
        if n<1 or n>64:
            raise RuntimeError('LGA original radius passed invalid feature sub-batch')
        if n<64:
            add=torch.arange(64-n,device=x.device)%n
            x=torch.cat([x,x.index_select(0,add)],dim=0)
        features=self.model.features(x)
        if features.ndim!=2 or features.shape[1]!=512:
            raise RuntimeError('LGA radius must operate on 512-d features')
        return features[:n]


def original_lga_radius(model,bank,accepted_train_indices,device,*,classes=(0,1)):
    """Author method called exactly, with only server-accepted training data."""
    from .upstream_glfc_lga_memory import _PrivateSourceImages
    if tuple(classes)!=(0,1):
        raise ValueError('LGA source radius is initialized on first two classes')
    accepted=set(int(i) for i in accepted_train_indices)
    provider=_PrivateSourceImages(bank.train,accepted)
    counts=[]
    for cls in classes:
        candidates=[i for i in accepted if int(bank.train.targets[i])==cls]
        if not candidates:
            raise RuntimeError(f'LGA radius cannot access unaccepted class {cls}')
        if len(candidates)%50:
            raise RuntimeError('Original LGA geometry uses 50-image source slices')
        counts.append(len(candidates))
    if counts[0]!=counts[1]:
        raise RuntimeError('LGA radius class counts not equal')
    original=original_lga_radius_class(cpu_test=device.type=='cpu')
    obj=original()
    obj.transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
    wrapped=_Strict64FeatureNetwork(model).to(device)
    original.compute_radius(obj,wrapped,provider,2,
                            (device.index if device.type=='cuda' else 0),
                            counts[0]//50)
    radius=float(obj.radius)
    if not np.isfinite(radius) or radius<0:
        raise RuntimeError('Original LGA radius is nonfinite/negative')
    return radius
