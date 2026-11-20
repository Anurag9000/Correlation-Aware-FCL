"""FEAT ECC inference correction: third-party reconstruction, NOT author code.

The official FEAT source lacks its imported utils/inc_net.py. The formula
below follows kiennkt05/FEAT's compatibility implementation. Do not cite
this module as an original-author source reproduction. Source projectors
and energy statistics are computed by the original FEAT geometry routines.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def paper_reconstructed_feat_logits(model,inputs,feat_ecc):
    if feat_ecc is None:
        return model(inputs)
    if getattr(model,'backbone_description',{}).get('model_id')!='native_feat_etf':
        raise RuntimeError('FEAT ECC needs the paper-reconstructed ETF classifier')
    features=model.features(inputs)
    if features.ndim!=2:raise ValueError('FEAT features must be [B,D]')
    d=features.shape[1]
    for key in ('PH','PT'):
        tensor=feat_ecc[key]
        if tuple(tensor.shape)!=(d,d):
            raise ValueError(f'FEAT {key} projector shape mismatch')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite FEAT {key} projector')
    energy=float(feat_ecc['energy']['bar_eH_G'])
    r_H=int(feat_ecc['r_H'])
    r_T=int(feat_ecc['r_T'])
    if not torch.isfinite(torch.tensor(energy)) or r_H<1 or r_T<1:
        raise ValueError('Invalid FEAT subspace energy/ranks')
    norm=F.normalize(features,dim=1)
    ph=feat_ecc['PH'].to(device=norm.device,dtype=norm.dtype)
    pt=feat_ecc['PT'].to(device=norm.device,dtype=norm.dtype)
    head=norm@ph
    tail=norm@pt
    energy_head=head.square().sum(dim=1)/r_H
    energy_tail=tail.square().sum(dim=1)/r_T
    gate=((energy_head-energy)/(energy_head+energy_tail+1e-8)).clamp_min(0)
    corrected=F.normalize(norm-gate[:,None]*head+gate[:,None]*tail,dim=1)
    return model.classifier(corrected)
