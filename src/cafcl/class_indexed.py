"""Single class-indexed output layer; local binary tasks and row-wise federation.

One output row is reserved per canonical dataset class ID. Rows are activated
by observed class labels; there are no task-specific heads. This preallocated
representation is deliberately shape-stable across disconnected clients.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


def classifier_parameter_names(model):
    """Return last Linear projection's parameter names for supported models."""
    found = [(n,m) for n,m in model.named_modules() if isinstance(m, nn.Linear)]
    if not found:
        raise ValueError('A class-indexed classifier requires a final Linear layer')
    name, layer=found[-1]
    prefix=f'{name}.' if name else ''
    return prefix+'weight', prefix+'bias' if layer.bias is not None else None


def canonical_pair_loss(logits, labels, classes):
    """Binary CE over exactly two GLOBAL class IDs, independent of local order."""
    pair=sorted(set(int(c) for c in classes))
    if len(pair)!=2 or pair[0]<0 or pair[-1]>=logits.shape[1]:
        raise ValueError('Every contrastive episode must have two distinct valid classes')
    ids=torch.as_tensor(pair,dtype=torch.long,device=logits.device)
    targets=labels.to(logits.device)
    allowed=(targets==pair[0])|(targets==pair[1])
    if not bool(torch.all(allowed)):
        raise ValueError('Current episode includes a label outside its declared pair')
    binary_targets=(targets==pair[1]).long()
    return F.cross_entropy(logits.index_select(1,ids),binary_targets)


def freeze_other_class_rows(model, active):
    """Keep every non-current output row fixed even after CL regularization/projection."""
    names=classifier_parameter_names(model)
    params=dict(model.named_parameters())
    rows=params[names[0]].shape[0]
    mask=torch.ones(rows,dtype=torch.bool,device=params[names[0]].device)
    mask[list(active)]=False
    for n in names:
        if n is not None and params[n].grad is not None:
            params[n].grad[mask]=0


def classwise_merge(base, candidate, uploads, previous_counts, classifier_keys, *, method="fedavg"):
    """Class-identity-safe row merge of uploaded and previously integrated heads.

    A client affects row c only if it actually trained c since its last sync.
    For ordinary weight averaging, use count-weighted old/updated rows; for
    adaptive aggregation the candidate delta is scaled by fresh class exposure.
    """
    out={k:v.clone() for k,v in candidate.items()}
    weight_key,bias_key=classifier_keys
    keys=[k for k in (weight_key,bias_key) if k is not None]
    for c in range(base[weight_key].shape[0]):
        old=max(0,int(previous_counts.get(c,0)))
        contributors=[(int(u.get('class_update_steps',{}).get(c,0)),u)
                      for u in uploads]
        contributors=[(n,u) for n,u in contributors if n>0]
        if not contributors:
            for key in keys:out[key][c]=base[key][c]
            continue
        # All algorithms have a class-aware accepted candidate. For explicit
        # weight averaging, use each uploader's class row, and the old server
        # exposure as a retained weighted contribution.
        fresh=sum(n for n,_ in contributors)
        for key in keys:
            if method in {'fedavg','uniform'}:
                numerator=base[key][c].to(torch.float64)*old
                for n,u in contributors:
                    numerator=numerator+n*u['weights'][key][c].to(torch.float64)
                out[key][c]=(numerator/(old+fresh)).to(base[key].dtype)
            else:
                # Preserve the underlying server optimizer/correction's update;
                # damp only by cumulative class-specific evidence.
                blend=fresh/(old+fresh)
                out[key][c]=(base[key][c].to(torch.float64)+blend*(
                    candidate[key][c].to(torch.float64)-base[key][c].to(torch.float64))).to(base[key].dtype)
    return out


def snapshot_frozen_rows(model, active):
    """Capture inactive class parameters to protect against SGD momentum drift."""
    names=classifier_parameter_names(model)
    params=dict(model.named_parameters())
    mask=torch.ones(params[names[0]].shape[0],dtype=torch.bool,device=params[names[0]].device)
    mask[list(active)]=False
    return mask,{n:params[n].detach()[mask].clone() for n in names if n is not None}


def restore_frozen_rows(model, optimizer, snapshot):
    """Restore exact row values and remove momentum on inactive outputs."""
    mask,old=snapshot
    params=dict(model.named_parameters())
    with torch.no_grad():
        for name,weight in old.items():
            param=params[name]
            param[mask]=weight
            state=optimizer.state.get(param,{})
            for value in state.values():
                if isinstance(value,torch.Tensor) and value.shape==param.shape:
                    value[mask]=0
