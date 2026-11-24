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
        # One model forward always contains 64 training examples. Padding repeats
        # eligible TRAIN inputs, but only the *original* selected examples enter
        # the diagonal-importance estimate; duplicate padding is never counted.
        for start in range(0,len(selected),64):
            batch_ids=selected[start:start+64]
            valid=len(batch_ids)
            padded=batch_ids+[batch_ids[i%valid] for i in range(64-valid)]
            input_x=torch.stack([dataset[int(i)][0] for i in padded]).to(device)
            logits=model(input_x)
            if len(logits)!=64:
                raise RuntimeError('EWC/MAS must forward a physical batch of 64')
            if active_classes is not None:
                classes=tuple(sorted(set(map(int,active_classes))))
                classifier=getattr(model,'classifier',None)
                if hasattr(classifier,'task_classes'):
                    # The active head already emits task-local columns; global
                    # class IDs must never be used as output column offsets.
                    actual=tuple(classifier.classes_for())
                    if not set(classes)<=set(actual):
                        raise ValueError('Importance objective exceeds active task classes')
                    columns=[actual.index(c) for c in classes]
                else:
                    columns=list(classes)
                active_ids=torch.as_tensor(columns,dtype=torch.long,device=logits.device)
                logits=logits.index_select(1,active_ids)
            if kind=='ewc' and logits.shape[1]<2:
                raise ValueError('Fisher requires >=2 logits')
            for local_index in range(valid):
                one=logits[local_index:local_index+1]
                if kind=='ewc':
                    probs=F.softmax(one.detach().to(torch.float64),dim=1)
                    label=torch.multinomial(probs.cpu(),1,generator=categorical_rng).to(one.device)
                    scalar=F.cross_entropy(one,label.reshape(-1))
                else:
                    scalar=one.square().sum()
                grads=torch.autograd.grad(scalar,params,allow_unused=True,
                                           retain_graph=(local_index+1<valid))
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
