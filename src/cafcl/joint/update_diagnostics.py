"""Parameter-only update alignment; BatchNorm buffers excluded."""
import math
from itertools import combinations
import torch


@torch.no_grad()
def update_geometry(model,base,uploads):
    """Applied candidate deltas relative to current global base.

    Stale candidate updates already include the declared rebasing/damping.
    Whole-model weighted cancellation is a reference, not the final
    task-head contributor-weighted aggregation rule.
    """
    if not uploads:return None
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    n=len(uploads);ids=[u['client_id'] for u in uploads]
    grams={k:torch.zeros((n,n),dtype=torch.float64) for k in ('backbone','classifier')}
    for name in names:
        if name not in base:raise ValueError('Missing parameter in base state')
        group='classifier' if name.startswith('classifier.') else 'backbone'
        # Chunked parameter-wise reduction; never concatenate entire models.
        reference=base[name].detach().cpu().reshape(-1)
        for start in range(0,reference.numel(),131072):
            end=min(start+131072,reference.numel())
            a=reference[start:end].double()
            d=torch.stack([u['weights'][name].detach().cpu().reshape(-1)[start:end].double()-a
                           for u in uploads])
            grams[group]+=d@d.T
    grams['all']=grams['backbone']+grams['classifier']
    weights=torch.tensor([u['examples'] for u in uploads],dtype=torch.float64)
    weights/=weights.sum()
    out={}
    for name,g in grams.items():
        norms=g.diagonal().clamp_min(0).sqrt()
        denominator=float(weights@norms)
        out[name]={'norms':{str(cid):float(norms[i]) for i,cid in enumerate(ids)},
            'cosines':{f'{ids[i]}:{ids[j]}':float(g[i,j]/(norms[i]*norms[j])) if norms[i]>0 and norms[j]>0 else None
                        for i,j in combinations(range(n),2)},
            'reference_cancellation_ratio':math.sqrt(max(0.,float(weights@g@weights)))/denominator if denominator else None}
    return out
