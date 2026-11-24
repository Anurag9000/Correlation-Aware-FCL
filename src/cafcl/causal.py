"""Aggregation-only causal diagnostics.

All candidate local updates must be trained from one identical global base version.
These analyses use validation results only and never feed them back to a
deployable selector; they are explicitly *privileged counterfactual diagnostics*.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import Iterable
import torch


def candidate_subsets(client_ids, size: int):
    """Enumerate fixed-cardinality cohorts; deterministic for auditing."""
    ids = tuple(sorted(set(client_ids)))
    if not 0 <= size <= len(ids):
        raise ValueError('Invalid subset size')
    return combinations(ids, size)


def pair_interaction(scores: dict[frozenset, float], base: Iterable[int], a: int, b: int):
    """Finite-difference interaction given four evaluated subset scores."""
    s = frozenset(base)
    if a in s or b in s or a == b:
        raise ValueError('Added clients must be distinct and absent from base')
    return (scores[s | {a,b}] - scores[s | {a}]
            - scores[s | {b}] + scores[s])


def aggregation_leave_one_out(model, bank, observed, updates, base_weights,
                              aggregator, batch_size, device, *, limit=6, task_class_groups=None):
    """Return per-class and overall effects of each uploaded client.

    The original full aggregate is compared with the aggregate omitting one
    update at a time, using the same held-out evaluation population. A single
    uploaded client has no valid leave-one-out FedAvg counterfactual.
    No client training is repeated here: this is *not* a full trajectory fork.
    """
    if len(updates) < 2 or len(updates) > limit:
        return {'kind': 'leave_one_out_aggregation', 'status': 'not_evaluated',
                'reason': 'requires 2..limit accepted updates'}
    from .simulator import aggregate, _merge_task_heads_fedavg
    from .evaluation import evaluate
    original = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    def score(chosen):
        candidate = aggregate(chosen, base_weights, aggregator)
        if task_class_groups is not None and aggregator=='fedavg':
            candidate=_merge_task_heads_fedavg(chosen,base_weights,candidate)
        model.load_state_dict(candidate)
        vals, _ = evaluate(model, bank, observed, batch_size, device, peak={},split='validation',task_class_groups=task_class_groups,loader_seed=0)
        return vals
    try:
        full = score(updates)
        impacts = {}
        for idx, item in enumerate(updates):
            smaller = score(updates[:idx]+updates[idx+1:])
            impacts[str(item['client_id'])] = {
                'accuracy_delta': full['accuracy'] - smaller['accuracy'],
                'per_class_delta': {str(c): full['per_class'][c] - smaller['per_class'][c]
                                    for c in full['per_class']},
                'omitted_accuracy': smaller['accuracy'],
            }
        return {'kind': 'leave_one_out_aggregation', 'status': 'evaluated',
                'full_accuracy': full['accuracy'], 'effects': impacts,
                'note': 'Retrospective oracle analysis; not a deployable client selector'}
    finally:
        model.load_state_dict(original)


def aggregation_subset_report(model,bank,observed,updates,base_weights,aggregator,
                              batch_size,device,*,limit=5,task_class_groups=None):
    """Evaluate every nonempty subset and pairwise interaction from SAME uploads.

    This is a retrospective *aggregation-only* oracle. It is intentionally
    bounded by `limit`, as it requires 2**K held-out evaluations and is not
    a feasible selection algorithm for large federations.
    """
    ids=[int(u['client_id']) for u in updates]
    if not 2<=len(ids)<=limit or len(set(ids))!=len(ids):
        return {'status':'not_evaluated','reason':'requires 2..limit unique uploads'}
    from .simulator import aggregate, _merge_task_heads_fedavg
    from .evaluation import evaluate
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    scores={}
    per_class={}
    try:
        # Empty subset is the unmodified previous global model.
        model.load_state_dict(base_weights)
        baseline,_=evaluate(model,bank,observed,batch_size,device,peak={},split='validation',task_class_groups=task_class_groups,loader_seed=0)
        scores[frozenset()]=baseline['accuracy']
        per_class[frozenset()]=baseline['per_class']
        for size in range(1,len(ids)+1):
            for chosen in combinations(ids,size):
                cohort=frozenset(chosen)
                selected=[u for u in updates if u['client_id'] in cohort]
                weights=aggregate(selected,base_weights,aggregator)
                if task_class_groups is not None and aggregator=='fedavg':
                    weights=_merge_task_heads_fedavg(selected,base_weights,weights)
                model.load_state_dict(weights)
                value,_=evaluate(model,bank,observed,batch_size,device,peak={},split='validation',task_class_groups=task_class_groups,loader_seed=0)
                scores[cohort]=value['accuracy']
                per_class[cohort]=value['per_class']
        interactions={}
        for a,b in combinations(sorted(ids),2):
            pair=frozenset((a,b))
            interactions[f'{a}:{b}']={
                'accuracy_interaction':pair_interaction(scores,(),a,b),
                'per_class_interaction':{
                    str(c):per_class[pair][c]-per_class[frozenset((a,))][c]
                    -per_class[frozenset((b,))][c]+per_class[frozenset()][c]
                    for c in baseline['per_class']}}
        return {'status':'evaluated','kind':'aggregation_subset_oracle',
                'evaluated_subsets':len(scores),
                'subset_accuracy':{','.join(map(str,sorted(key))):value
                                   for key,value in scores.items()},
                'pair_interactions':interactions,
                'note':'Diagnostic only. No additional local client training was executed.'}
    finally:
        model.load_state_dict(original)
