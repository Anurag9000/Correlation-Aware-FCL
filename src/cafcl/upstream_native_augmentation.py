"""Directly compile official GLFC/LGA fl_main.py CIFAR100 PIL transforms.

The user requires full physical batch64 and 5 independent client streams,
so only the data source & deterministic per-client RNG are adapted.
Each transform object is created by executing the author's original AST
assignment from fl_main.py, not a generic replacement or FEAT transform.
"""
from __future__ import annotations
import ast
import random
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2]
SOURCES={'native_glfc':'GLFC/src/fl_main.py',
         'native_lga':'LGA/fl_main.py'}
_RNG_LOCK=RLock()


@lru_cache(maxsize=None)
def original_glfc_lga_train_transform(method):
    if method not in SOURCES:raise ValueError(method)
    path=ROOT/SOURCES[method]
    source=ast.parse(path.read_text(),filename=str(path))
    nodes=[node for node in source.body
           if isinstance(node,ast.Assign) and
           any(isinstance(name,ast.Name) and name.id=='train_transform'
               for name in node.targets)]
    if len(nodes)!=1:
        raise RuntimeError('Authors source training transform missing')
    module=ast.Module(body=nodes,type_ignores=[])
    ast.fix_missing_locations(module)
    env={'args':SimpleNamespace(img_size=32),'transforms':transforms}
    exec(compile(module,str(path),'exec'),env)
    transform=env['train_transform']
    types=[type(t).__name__ for t in transform.transforms]
    if types!=['RandomCrop','RandomHorizontalFlip','ColorJitter',
               'ToTensor','Normalize']:
        raise RuntimeError(f'Unrecognized original {method} transform: {types}')
    return transform


def original_glfc_lga_augment_batch(method,x,seed):
    """Run author PIL crop/flip/brightness transform on normalized source bytes."""
    if tuple(x.shape[1:])!=(3,32,32) or x.ndim!=4:
        raise ValueError('Author CIFAR transform expects [N,3,32,32]')
    if x.device.type!='cpu':
        raise RuntimeError('Authors torchvision PIL pipeline is CPU-only')
    from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
    mean=torch.tensor(NORMALIZATION_MEAN,dtype=x.dtype).view(1,3,1,1)
    std=torch.tensor(NORMALIZATION_STD,dtype=x.dtype).view(1,3,1,1)
    pixels=((x.detach()*std+mean).clamp(0,1)*255).round().byte()
    transform=original_glfc_lga_train_transform(method)
    with _RNG_LOCK:
        oldtorch=torch.get_rng_state()
        oldpython=random.getstate()
        try:
            gen=torch.Generator(device='cpu').manual_seed(int(seed))
            torch.set_rng_state(gen.get_state())
            random.seed(int(seed))
            result=[transform(Image.fromarray(
                    raw.permute(1,2,0).contiguous().numpy(),'RGB'))
                    for raw in pixels]
        finally:
            torch.set_rng_state(oldtorch)
            random.setstate(oldpython)
    return torch.stack(result)


def entropy_source_collate(method,seed):
    """Source entropy's sampled TRAIN loader, with original crop-flip-jitter."""
    calls=[0]
    def collect(batch):
        ids=torch.tensor([int(s[0]) for s in batch])
        xs=torch.stack([s[1] for s in batch])
        labels=torch.tensor([int(s[2]) for s in batch])
        xs=original_glfc_lga_augment_batch(method,xs,
              seed=seed+calls[0]*997)
        calls[0]+=1
        return ids,xs,labels
    return collect
