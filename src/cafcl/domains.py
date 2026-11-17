"""Explicit, deterministic image-domain transformations and evaluation slices.

A domain observation is the pair (client, stage), not merely a label.
Test images are never reused as training samples, and each observed client-domain
slice remains in the full-population evaluation bank, even if inactive.
"""
from __future__ import annotations
from torch.utils.data import Dataset
import torch


def transform_domain(x:torch.Tensor,client_id:int,stage:int,mode:str='domain'):
    """Fixed label-preserving CIFAR transformations with observable domain drift.

    Stage zero is the base domain; subsequent stages rotate/flip images or
    induce contrast and color shifts. Inverted-color and stronger shifts can
    make labels harder, so this is a controlled robustness setting, not a
    claim of naturally collected domain data.
    """
    if stage<=0:return x
    t=(stage+client_id)%4
    if mode=='gradual_drift':
        severity=min(.7,stage*.07)
        return x*(1.-severity)+severity*x.flip(-1)
    if t==0:return torch.flip(x,(-1,))
    if t==1:return x*1.15+.1
    if t==2:return torch.rot90(x,1,(-2,-1))
    return x.flip(-2)*.85-.1


class ClientDomainDataset(Dataset):
    def __init__(self,base,client_id,sample_stages,mode='domain'):
        self.base=base
        self.client_id=int(client_id)
        self.sample_stages=sample_stages
        self.mode=mode
        self.targets=base.targets
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        return transform_domain(x,self.client_id,self.sample_stages.get(i,0),self.mode),y


class GlobalDomainTest(Dataset):
    """Held-out test over all *arrived* client-domain-label slices.

    Sample identity is (client_id, experience_stage, original_test_index).
    Per-client/slice duplication is intentional because the distribution
    (domain transform) differs; the official test pool itself stays held out.
    """
    def __init__(self,base,observed_slices,mode='domain'):
        self.base=base
        self.mode=mode
        self.items=[]
        for cid,stage,labels in observed_slices:
            allowed=set(map(int,labels))
            self.items.extend((int(cid),int(stage),i)
                              for i,y in enumerate(base.targets) if int(y) in allowed)
        self.targets=[int(base.targets[idx]) for _,_,idx in self.items]
        self.slice_keys=[(cid,stage) for cid,stage,_ in self.items]
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        cid,stage,idx=self.items[i]
        x,y=self.base[idx]
        return transform_domain(x,cid,stage,self.mode),y


def apply_spurious_patch(x:torch.Tensor,y:int,strength:float,reverse=False):
    """Deliberately label-correlated top-left marker (train-only generator).

    Test variants use the same data-generating rule or its reversed version;
    the classifier is never given the label at inference, only the image.
    """
    if strength<=0:return x
    result=x.clone()
    bit=(int(y)%2==0) != bool(reverse)
    value=strength if bit else -strength
    result[...,0:4,0:4]=value
    return result


class ClientAugmentedDataset(Dataset):
    """Deterministic client-domain transformation and optional training noise."""
    def __init__(self,base,client_id,sample_stages,mode='disjoint',*,
                 spurious_strength=0.,label_noise_rate=0.,seed=0,num_classes=10):
        self.base=base
        self.targets=base.targets
        self.client_id=int(client_id)
        self.sample_stages=sample_stages
        self.mode=mode
        self.spurious_strength=float(spurious_strength)
        self.label_noise_rate=float(label_noise_rate)
        self.seed=int(seed)
        self.num_classes=int(num_classes)
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        if self.mode in {'domain','gradual_drift','shifted_recurrence'}:
            x=transform_domain(x,self.client_id,self.sample_stages.get(i,0),self.mode)
        x=apply_spurious_patch(x,y,self.spurious_strength)
        if self.label_noise_rate:
            # Seeded from stable sample/client IDs, not iteration order.
            import numpy as np
            rng=np.random.default_rng((self.seed*1000003+self.client_id*1009+int(i))%(2**32))
            if rng.random()<self.label_noise_rate:
                offset=int(rng.integers(1,self.num_classes))
                y=(int(y)+offset)%self.num_classes
        return x,int(y)


class SpuriousTest(Dataset):
    def __init__(self,base,strength,reverse=False):
        self.base=base
        self.targets=base.targets
        self.strength=strength
        self.reverse=reverse
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        return apply_spurious_patch(x,y,self.strength,self.reverse),y
