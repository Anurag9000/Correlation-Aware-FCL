"""Native-paper-inspired loss adapters for the frozen GLFC, LGA and FEAT snapshots.

The original upstream trees are unchanged. This is a protocol-harmonized port,
not a bitwise replication: the source GLFC/LGA global-class and proxy assumptions
do not support independent sparse class streams without adaptation.
"""
from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from .upstream_losses import original_loss,original_feat_gsa

NATIVE_METHODS=frozenset({"native_glfc","native_lga","native_feat"})


def _subset(logits,labels,classes):
    ids=torch.tensor(sorted(set(map(int,classes))),device=logits.device,dtype=torch.long)
    inverse=torch.full((logits.shape[1],),-1,device=logits.device,dtype=torch.long)
    inverse[ids]=torch.arange(ids.numel(),device=logits.device)
    mapped=inverse[labels]
    if (mapped<0).any():
        raise RuntimeError("Native objective received an unseen label")
    return logits.index_select(1,ids),mapped,ids


def weighted_bce(logits,labels,classes,old_classes,teacher_logits=None,method="native_glfc"):
    """Preserves the weighted BCE + old-model sigmoid distillation of GLFC/LGA.

    Sparse global IDs are gathered to the client's actual known-class columns.
    No future-class outputs or task-ID oracle may enter the local objective.
    """
    z,y,ids=_subset(logits,labels,classes)
    target=F.one_hot(y,ids.numel()).to(z.dtype)
    g=(torch.sigmoid(z.detach())-target).abs().gather(1,y[:,None])
    old_mask=torch.zeros_like(y,dtype=torch.bool)
    if old_classes:
        old_mask=torch.isin(labels,torch.tensor(sorted(old_classes),device=labels.device))
    new_mask=~old_mask
    weight=torch.zeros_like(g)
    for mask in (old_mask,new_mask):
        if mask.any():
            group=g[mask]
            weight[mask]=group/(group.mean()+1e-12)
    current=(weight*F.binary_cross_entropy_with_logits(z,target,reduction="none")).mean()
    if teacher_logits is None or not old_classes:
        return current
    historical=sorted(set(old_classes)&set(classes))
    if not historical:
        return current
    t=z.new_zeros(z.shape)
    t.copy_(target)
    col={int(c):i for i,c in enumerate(ids.tolist())}
    old_idx=torch.tensor([col[c] for c in historical],device=z.device)
    teacher_idx=torch.tensor(historical,device=z.device)
    t[:,old_idx]=teacher_logits.detach().index_select(1,teacher_idx).sigmoid()
    distill=F.binary_cross_entropy_with_logits(z,t,reduction="none")
    if method=="native_glfc":
        # GLFC.py: original 0.5 * weighted BCE + 0.5 * BCE distillation.
        return .5*current+.5*distill.mean()
    if method=="native_lga":
        # LGA.py: default lam=1.0; both terms use the sample weights.
        return current+(weight*distill).mean()
    raise ValueError(method)


def feat_gsa(features,labels,weight,known_classes,tau=.07,eps=1e-8):
    """Run original FEAT GSA math, with sparse-global-ID and FC shape adapter.

    Published source itself is compiled directly from frozen FEAT/methods/FEAT.py.
    The original ETF classifier is still unavailable from the published repo.
    """
    return original_feat_gsa(features,labels,weight,known_classes,tau=tau,eps=eps)


def native_loss(method,model,inputs,labels,known_classes,teacher=None,old_classes=(),feat_kd=.1):
    """One optimizer forward on exactly the supplied physical batch."""
    if method not in NATIVE_METHODS:raise ValueError(method)
    if not hasattr(model,"features") or not hasattr(model,"classifier"):
        raise RuntimeError("Native model must expose feature extractor and classifier")
    features=model.features(inputs)
    logits=model.classifier(features)
    if method=="native_feat":
        z,y,_=_subset(logits,labels,known_classes)
        loss=F.cross_entropy(z,y)+feat_kd*feat_gsa(
            features,labels,model.classifier.weight,known_classes)
        return loss,logits
    # Execute the exact authors' _compute_loss/efficient_old_class_weight
    # methods, with only sparse global label -> contiguous author-label mapping.
    # Other original server/proxy/exemplar mechanisms are still not restored;
    # the pilot stays blocked from claiming complete method fidelity.
    return original_loss(method,logits,labels,known_classes,
                         old_classes=old_classes,
                         old_teacher=teacher,inputs=inputs),logits


def historical_teacher(model,state,device):
    if state is None:return None
    teacher=copy.deepcopy(model)
    teacher.load_state_dict(state)
    teacher.to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return teacher
