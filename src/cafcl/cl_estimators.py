"""Paper-grounded per-example diagonal-importance estimators for EWC and MAS.

Neither estimator reads validation/test samples. The bounded number of TRAIN
examples controls computational expense, not a mathematically incorrect
minibatch-mean-gradient-square shortcut.

EWC: model-predictive diagonal Fisher, E_{x ~ TRAIN, y ~ p_theta(.|x)}
      [(d/dtheta log p_theta(y|x))^2], approximated by one model-drawn
      class per sampled input.
MAS: mean over TRAIN examples of |d/dtheta ||f_theta(x)||_2^2|.
      With strict class-pair tasks, the task's active output subspace is used.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def estimate_diagonal_importance(
    model, dataset, indices, device, *,
    kind, max_samples, seed, active_classes=None
):
    if kind not in {'ewc','mas'}:
        raise ValueError(f'Unknown importance estimator: {kind}')
    if max_samples < 1:
        raise ValueError('Importance sample count must be positive')
    unique=list(dict.fromkeys(map(int,indices)))
    if not unique:
        raise ValueError('Cannot estimate importance on empty experience')
    rng=np.random.default_rng(int(seed))
    if len(unique)>max_samples:
        selected=[unique[int(j)] for j in rng.choice(len(unique),size=max_samples,replace=False)]
    else:
        selected=unique
    named=dict(model.named_parameters())
    params=list(named.values())
    result={name:torch.zeros_like(p.detach(),dtype=torch.float64,device='cpu')
            for name,p in named.items()}
    was_training=model.training
    # These importance computations must not update BN buffers or dropout RNG;
    # they are read-only measurements made after the local task is trained.
    model.eval()
    categorical_rng=torch.Generator(device='cpu').manual_seed(int(seed)+9173)
    try:
        for index in selected:
            x,_=dataset[index]
            logits=model(x.unsqueeze(0).to(device))
            if active_classes is not None:
                ids=torch.as_tensor(sorted(set(map(int,active_classes))),
                    dtype=torch.long,device=logits.device)
                logits=logits.index_select(1,ids)
            if kind=='ewc':
                if logits.shape[1]<2:
                    raise ValueError('Fisher calculation requires >=2 classifier outputs')
                prob=F.softmax(logits.detach().to(torch.float64),dim=1)
                # Sampling a label from the model predictive distribution
                # distinguishes true Fisher from empirical-label Fisher.
                label=torch.multinomial(prob.cpu(),1,generator=categorical_rng).to(logits.device)
                scalar=F.cross_entropy(logits,label.reshape(-1))
            else:
                # MAS is label-independent. Take the sensitivity for each
                # *individual* example, then the absolute value, then average.
                scalar=logits.square().sum(dim=1).sum()
            grads=torch.autograd.grad(scalar,params,allow_unused=True)
            for (name,_),g in zip(named.items(),grads):
                if g is not None:
                    contribution=g.detach().cpu().to(torch.float64)
                    if kind=='ewc':
                        contribution=contribution.square()
                    else:
                        contribution=contribution.abs()
                    result[name]+=contribution
        for name in result:
            result[name]=(result[name]/len(selected)).to(named[name].dtype)
            if not torch.isfinite(result[name]).all():
                raise RuntimeError(f'Nonfinite {kind} diagonal importance in {name}')
        return result
    finally:
        model.train(was_training)
