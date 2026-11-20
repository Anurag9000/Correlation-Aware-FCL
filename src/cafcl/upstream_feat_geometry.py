"""Execute original FEAT ECC projection / energy-prior code, without rewriting.

Original source is fixed to FEAT/methods/FEAT.py. This adapts only the
global sparse-class model, per-client TRAIN stream and historical replay
to FEAT's original compact-class classifier and dataloader.

This module does NOT supply the missing author ETF classifier or inference
correction; an ECC-prior receipt alone is not a faithful FEAT reproduction.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[2]
FUNCTIONS=('_build_projector_from_W','estimate_ecc_priors','aggregate_ecc_priors_from_clients')


@lru_cache(maxsize=1)
def original_feat_geometry():
    """Return the three authors' exact source function objects."""
    path=ROOT/'FEAT/methods/FEAT.py'
    source=ast.parse(path.read_text(),filename=str(path))
    methods=[n for n in source.body
             if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    if {n.name for n in methods}!=set(FUNCTIONS):
        raise RuntimeError('Official FEAT geometric prior methods missing')
    module=ast.Module(body=methods,type_ignores=[])
    ast.fix_missing_locations(module)
    env={'torch':torch,'F':F}
    exec(compile(module,str(path),'exec'),env)
    return {name:env[name] for name in FUNCTIONS}


class _CompactFEATView(nn.Module):
    def __init__(self,model,classes):
        super().__init__()
        self.model=model
        ids=torch.as_tensor(classes,device=model.classifier.weight.device,dtype=torch.long)
        # Source estimate_ecc_priors reads self.fc.weight [C,d] but
        # computes features through model(images)['features'].
        self.fc=SimpleNamespace(weight=model.classifier.weight.index_select(0,ids))
    def forward(self,inputs):
        return {'features':self.model.features(inputs)}


class _AvailableExamples(Dataset):
    def __init__(self,data,indices,classes,targets):
        self.data=data
        self.indices=tuple(dict.fromkeys(map(int,indices)))
        self.reverse={int(c):int(i) for i,c in enumerate(classes)}
        self.targets=targets
        if any(int(targets[i]) not in self.reverse for i in self.indices):
            raise ValueError('FEAT geometric prior received an unexposed class')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        idx=self.indices[i]
        x,y=self.data[idx]
        if int(y)!=int(self.targets[idx]):
            raise RuntimeError('FEAT sample bytes/labels mismatch')
        return idx,x,self.reverse[int(y)]


def compute_client_feat_ecc(model,client,bank,device,active_pair,*,batch_size=64,ema_decay=.9):
    """Compute source ECC priors for one client's actually available examples.

    Returns None when there are no old classes (source itself cannot
    construct the old-class projector in that case). Does not access
    hidden peer images, future labels, or test/validation data.
    """
    from .federated_cl import ClientArchiveDataset
    seen=tuple(sorted(map(int,client.observed_classes)))
    new=tuple(sorted(set(map(int,active_pair or ()))))
    old=tuple(c for c in seen if c not in new)
    if not new or not old:return None
    if set(new)-set(seen):
        raise RuntimeError('FEAT prior requested future classes')
    pool=[int(i) for group in client.memory.values() for i in group]
    current=list(map(int,client.current_indices))
    ids=list(dict.fromkeys(current+pool))
    for idx in ids:
        if idx not in client.seen_indices and idx not in client.received_example_payloads:
            raise RuntimeError('FEAT prior tried to access unsynchronized remote data')
    source=(ClientArchiveDataset(bank.train,client.received_example_payloads)
            if client.received_example_payloads else bank.train)
    dataset=_AvailableExamples(source,ids,seen,bank.train.targets)
    if not len(dataset):return None
    # The user's 64-input requirement is enforced by the same sampler as
    # the experimental local training. Tail padding is a documented
    # user-protocol change to source analytical sample weighting.
    from .continual import _FixedBatchSampler
    sampler=_FixedBatchSampler(len(dataset),batch_size,
                                seed=101+client.client_id+client.local_experience*10009,
                                shuffle=False)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0)
    original=original_feat_geometry()
    compact=_CompactFEATView(model,seen).to(device)
    mapping={c:i for i,c in enumerate(seen)}
    priors,ph,pt=original['estimate_ecc_priors'](
        compact,loader,[mapping[c] for c in new],[mapping[c] for c in old],
        ema_decay=ema_decay,feature_key='features')
    if ph is None or pt is None:return None
    if int(priors['nT'])<1:return None
    if not (torch.isfinite(ph).all() and torch.isfinite(pt).all()):
        raise RuntimeError('Nonfinite original FEAT projector')
    return dict(
        priors={k:float(v) if k!='nT' else int(v) for k,v in priors.items()},
        PH=ph.detach().cpu().clone(),PT=pt.detach().cpu().clone(),
        r_H=max(len(new)-1,1),r_T=max(len(old)-1,1),
        new_classes=list(new),old_classes=list(old),
        source='FEAT/methods/FEAT.py:estimate_ecc_priors',
        client=int(client.client_id),
        local_experience=int(client.local_experience),
    )


def consolidate_original_ecc(receipts):
    """Author aggregate_ecc_priors_from_clients + source projection payload.

    The authors aggregate scalar energies across selected clients and
    retain the last selected client's P_H/P_T (source FEAT._fl_train).
    Preserve that behavior as a source-specific choice; order receipts by
    client id and local experience to make it deterministic.
    """
    if not receipts:return None
    ordered=sorted(receipts,key=lambda r:(int(r['client']),int(r['local_experience'])))
    aggregate=original_feat_geometry()['aggregate_ecc_priors_from_clients'](
        [r['priors'] for r in ordered],eps=1e-8)
    if aggregate is None:return None
    latest=ordered[-1]
    return dict(
        energy=dict(aggregate),
        PH=latest['PH'].clone(),PT=latest['PT'].clone(),
        r_H=int(latest['r_H']),r_T=int(latest['r_T']),
        projector_source_client=int(latest['client']),
        accepted_clients=sorted({int(r['client']) for r in ordered}),
        receipt_count=len(ordered),
        source='FEAT/methods/FEAT.py:aggregate_ecc_priors_from_clients',
    )


def clone_feat_ecc(obj):
    if obj is None:return None
    return {key:(val.detach().cpu().clone() if isinstance(val,torch.Tensor)
                 else dict(val) if isinstance(val,dict)
                 else list(val) if isinstance(val,list)
                 else val)
            for key,val in obj.items()}
