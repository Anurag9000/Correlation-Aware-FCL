"""Original FEAT CIFAR-100 RandomCrop/Flip/Brightness author transform.

Compiles the unchanged FEAT._get_norm_and_transform and Normalizer
definitions from the pinned authors' FEAT/methods/FEAT.py, then applies
its returned torchvision transform on recovered exact uint8 images.
No validation or test images are augmented.

A deterministic per-client CPU RNG scope is a necessary concurrency
adaptation because the original trainer loops over clients sequentially,
whereas the experiment runs them in concurrent task-barrier workers.
"""
from __future__ import annotations
import ast
from functools import lru_cache
from pathlib import Path
from threading import Lock
import random

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
_RNG_LOCK=Lock()


@lru_cache(maxsize=1)
def original_feat_cifar100_transform():
    path=ROOT/'FEAT/methods/FEAT.py'
    parsed=ast.parse(path.read_text(),filename=str(path))
    classes={n.name:n for n in parsed.body if isinstance(n,ast.ClassDef)}
    if 'Normalizer' not in classes or 'FEAT' not in classes:
        raise RuntimeError('FEAT author normalization/learner absent')
    methods=[n for n in classes['FEAT'].body
             if isinstance(n,ast.FunctionDef) and n.name=='_get_norm_and_transform']
    if len(methods)!=1:
        raise RuntimeError('Author FEAT transform selector missing')
    isolated=ast.Module(body=[
        classes['Normalizer'],methods[0]],type_ignores=[])
    ast.fix_missing_locations(isolated)
    env={'torch':torch,'transforms':transforms}
    exec(compile(isolated,str(path),'exec'),env)
    selector=env['_get_norm_and_transform']
    transform,_=selector(None,'cifar100')
    if not isinstance(transform,transforms.Compose):
        raise RuntimeError('Unexpected original FEAT transformation')
    names=[type(f).__name__ for f in transform.transforms]
    assert names==['RandomCrop','RandomHorizontalFlip','ColorJitter',
                  'ToTensor','Normalize'],names
    return transform


def original_feat_augment_batch(batch,seed):
    """Apply author CIFAR100 PIL transforms to a CPU [N,3,32,32] tensor.

    Input data are the standard normalized CIFAR100 bytes. Reverse the
    original affine normalization and round to uint8: CIFAR100's initial
    ToTensor/Normalize is exactly reversible to source pixel precision.
    """
    if batch.ndim!=4 or tuple(batch.shape[1:])!=(3,32,32):
        raise ValueError('Original FEAT augmentation expects [N,3,32,32]')
    if batch.device.type!='cpu':
        raise RuntimeError('Author FEAT PIL augmentation is CPU-side')
    transform=original_feat_cifar100_transform()
    mean=torch.tensor((.5071,.4867,.4408),dtype=batch.dtype).view(1,3,1,1)
    std=torch.tensor((.2675,.2565,.2761),dtype=batch.dtype).view(1,3,1,1)
    original=((batch.detach()*std+mean).clamp(0,1)*255).round().to(torch.uint8)
    outputs=[]
    # RandomCrop/HorizontalFlip/ColorJitter use PyTorch CPU global RNG.
    # The lock avoids interleaving multiple FEAT source transformations.
    with _RNG_LOCK:
        previous_torch=torch.get_rng_state()
        previous_python=random.getstate()
        try:
            generator=torch.Generator(device='cpu').manual_seed(int(seed))
            torch.set_rng_state(generator.get_state())
            random.seed(int(seed))
            for pixels in original:
                image=Image.fromarray(pixels.permute(1,2,0).contiguous().numpy(),'RGB')
                outputs.append(transform(image))
        finally:
            torch.set_rng_state(previous_torch)
            random.setstate(previous_python)
    return torch.stack(outputs,dim=0)
