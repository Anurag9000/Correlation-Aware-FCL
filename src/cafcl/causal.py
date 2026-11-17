"""Aggregation-only causal diagnostics.

All candidate local updates must be trained from one identical global base version.
These analyses never feed the held-out evaluation results back to a deployable
selector; they are explicitly *privileged counterfactual diagnostics*.
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
                              aggregator, batch_size, device, *, limit=6):
    """Return per-class and overall effects of each uploaded client.

    The original full aggregate is compared with the aggregate omitting one
    update at a time, using the same held-out evaluation population. A single
    uploaded client has no valid leave-one-out FedAvg counterfactual.
    No client training is repeated here: this is *not* a full trajectory fork.
    """
    if len(updates) < 2 or len(updates) > limit:
        return {'kind': 'leave_one_out_aggregation', 'status': 'not_evaluated',
                'reason': 'requires 2..limit accepted updates'}
    from .simulator import aggregate
    from .evaluation import evaluate
    original = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    def score(chosen):
        candidate = aggregate(chosen, base_weights, aggregator)
        model.load_state_dict(candidate)
        vals, _ = evaluate(model, bank, observed, batch_size, device, peak={})
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
