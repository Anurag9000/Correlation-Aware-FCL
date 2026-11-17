"""Privileged offline class-relationship and gradient diagnostics.

These routines can inspect the simulator's *training* bank to explain model
behavior. They must never be silently used in a privacy-preserving online
client-selection method or fit on held-out test examples.
"""
from __future__ import annotations
from collections import defaultdict
import itertools
import math
import numpy as np
import torch
from torch.nn import functional as F


def vector_cosine(left, right, eps=1e-12):
    a=left.detach().reshape(-1).to(dtype=torch.float64)
    b=right.detach().reshape(-1).to(dtype=torch.float64)
    if a.numel()!=b.numel():
        raise ValueError('Expected equal-length vectors')
    denominator=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    if denominator.item()<=eps:
        return None
    return float((torch.dot(a,b)/denominator).item())


def label_jaccard(left, right):
    a,b=set(left),set(right)
    union=a|b
    return len(a&b)/len(union) if union else 1.0


def realized_client_overlap(scenario):
    holders=defaultdict(set)
    for a in scenario.arrivals:
        for c in a.labels:
            holders[int(a.client_id)].add(int(c))
    return {f'{i}:{j}':label_jaccard(holders[i],holders[j])
            for i,j in itertools.combinations(range(scenario.num_clients),2)}


def lagged_binary_arrival_correlation(scenario,class_a,class_b,lag=0):
    """A finite-window correlation of aggregate arrivals, not causal evidence."""
    if lag<0:raise ValueError('Lag must be nonnegative')
    horizon=1+max(a.round_id for a in scenario.arrivals)
    a=np.zeros(horizon,dtype=np.float64)
    b=np.zeros(horizon,dtype=np.float64)
    for e in scenario.arrivals:
        if class_a in e.labels:a[e.round_id]+=1
        if class_b in e.labels:b[e.round_id]+=1
    if lag>=horizon:return None
    if lag:
        a=a[:-lag];b=b[lag:]
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def feature_class_means(model, bank, class_ids, device, samples_per_class=32):
    """Compute means from training samples with deterministic sample selection."""
    if samples_per_class<=0:raise ValueError('samples_per_class must be positive')
    model.eval()
    means={}
    with torch.no_grad():
        for cls in sorted(set(map(int,class_ids))):
            ids=[i for i,c in enumerate(bank.train.targets) if int(c)==cls][:samples_per_class]
            if not ids:continue
            x=torch.stack([bank.train[i][0] for i in ids]).to(device)
            if hasattr(model,'features'):
                features=model.features(x)
            else:
                features=model(x)
            means[cls]=features.float().mean(0).detach().cpu()
    return means


def feature_relationships(model,bank,class_ids,device,samples_per_class=32):
    means=feature_class_means(model,bank,class_ids,device,samples_per_class)
    return {f'{a}:{b}':vector_cosine(means[a],means[b])
            for a,b in itertools.combinations(sorted(means),2)}


def diagnostic_gradient_relationships(model,bank,class_ids,device,samples_per_class=8,
                                      max_pairs=32):
    """Pairwise class-loss gradient cosine at one common model checkpoint.

    The input labels and sample indices are drawn from the training bank;
    the user-visible test set is never used to choose gradient diagnostics.
    Because the simulator controls all client data, this is an *oracle
    diagnostic* and should not be a server input in privacy-constrained runs.
    """
    if samples_per_class<=0 or max_pairs<0:
        raise ValueError('Invalid diagnostic limit')
    model.eval()
    params=[p for p in model.parameters() if p.requires_grad]
    vectors={}
    for cls in sorted(set(map(int,class_ids))):
        ids=[i for i,y in enumerate(bank.train.targets) if int(y)==cls][:samples_per_class]
        if not ids:continue
        x=torch.stack([bank.train[i][0] for i in ids]).to(device)
        y=torch.tensor([cls]*len(ids),device=device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x),y).backward()
        vectors[cls]=torch.cat([(p.grad.detach() if p.grad is not None
                                else torch.zeros_like(p)).reshape(-1).cpu()
                               for p in params])
    model.zero_grad(set_to_none=True)
    return {f'{a}:{b}':vector_cosine(vectors[a],vectors[b])
            for a,b in itertools.islice(itertools.combinations(sorted(vectors),2),max_pairs)}


def class_gradient_representations(model,bank,class_ids,device,*,allowed_indices=None,samples_per_class=8):
    """Privileged TRAIN-only class gradients at one global checkpoint.

    No future samples are eligible: caller supplies indices that have actually
    arrived somewhere in the federation. Never use held-out evaluation data.
    Intended only for diagnostics and labeled oracle client selection.
    """
    allowed=set(allowed_indices) if allowed_indices is not None else set(range(len(bank.train)))
    vectors={}
    params=[p for p in model.parameters() if p.requires_grad]
    previous_training=model.training
    model.eval()
    for cls in sorted(set(map(int,class_ids))):
        ids=[i for i,y in enumerate(bank.train.targets)
             if i in allowed and int(y)==cls][:samples_per_class]
        if not ids:continue
        x=torch.stack([bank.train[i][0] for i in ids]).to(device)
        y=torch.tensor([cls]*len(ids),device=device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x),y).backward()
        vectors[cls]=torch.cat([(p.grad.detach().reshape(-1).cpu() if p.grad is not None
                                  else torch.zeros_like(p).reshape(-1).cpu()) for p in params])
    model.zero_grad(set_to_none=True)
    model.train(previous_training)
    return vectors


def arrived_feature_means(model,bank,class_ids,device,*,allowed_indices,samples_per_class=16):
    """Feature prototypes from already-arrived training examples only."""
    allowed=set(allowed_indices)
    means={}
    previous_training=model.training
    model.eval()
    with torch.no_grad():
        for cls in sorted(set(map(int,class_ids))):
            ids=[i for i,y in enumerate(bank.train.targets)
                 if i in allowed and int(y)==cls][:samples_per_class]
            if not ids:continue
            x=torch.stack([bank.train[i][0] for i in ids]).to(device)
            embeddings=model.features(x) if hasattr(model,'features') else model(x)
            means[cls]=embeddings.detach().float().mean(dim=0).cpu()
    model.train(previous_training)
    return means


def pairwise_vector_cosines(vectors):
    return {(a,b):vector_cosine(vectors[a],vectors[b])
            for a,b in itertools.combinations(sorted(vectors),2)}
