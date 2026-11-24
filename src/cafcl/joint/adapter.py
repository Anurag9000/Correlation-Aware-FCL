"""Validated on-disk joint plan -> existing Scenario adapter.

Only the original Replay/SGD + FedAvg engine is enabled in this initial
runtime path. A compiled plan, not its generator, determines training.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from ..data import Arrival, Scenario
from .curriculum import compile_plan, digest


@dataclass
class JointScenario(Scenario):
    joint_plan: dict | None = None


def load_plan(path, expected_hash, train_targets):
    saved=json.loads(Path(path).read_text())
    claimed=saved.get('plan_hash')
    payload={k:v for k,v in saved.items() if k!='plan_hash'}
    if claimed!=expected_hash or digest(payload)!=claimed:
        raise ValueError('Joint plan hash mismatch')
    if digest(list(map(int,train_targets)))!=saved['allocation_audit']['training_target_hash']:
        raise ValueError('Joint plan is not for this TRAIN split')
    # Reconstruct the deterministic plan and compare all fields. This checks
    # sample labels, allocation, coordinates, IDs, coverage and all boundaries.
    tasks=saved['tasks']
    learning=[r for r in saved['intervals'] if not r['terminal_flush']]
    slicing=saved['allocation_audit'].get('epoch_slicing')
    rebuilt=compile_plan(
        slicing['parent_events'] if slicing else [(x['atomic_step'],x['client_id'],x['classes']) for x in tasks],
        train_targets,num_clients=saved['num_clients'],num_classes=saved['num_classes'],
        seed=saved['seed'],head_policy=saved['head_policy'],
        require_all_classes=False,
        interval_boundaries=None if slicing else [max(r['atomic_steps'])+1 for r in learning],
        upload_cohorts=None if slicing else [r['upload_clients'] for r in learning],
        terminal_flush=saved['intervals'][-1]['terminal_flush'],
        allocation_order=saved['allocation_audit'].get('allocation_order','timeline'),
        raw_sample_overlap=saved['allocation_audit']['raw_sample_overlap'],
        allow_cross_semantic_reuse=saved['allocation_audit'].get('allow_cross_semantic_reuse',False),
        allocation_weights=saved['allocation_audit'].get('allocation_weights'),
        minimum_samples=saved['allocation_audit'].get('allocation_minimum_samples',1),
        control_kind=saved['allocation_audit'].get('control_kind','federated'),
        epoch_segments=slicing['segments'] if slicing else 1)
    if rebuilt.to_dict()['plan_hash']!=claimed:
        raise ValueError('Joint plan reconstruction differs from saved plan')
    return saved


def compile_joint_scenario(bank,cfg):
    plan=load_plan(cfg.joint_plan_path,cfg.joint_plan_hash,bank.train.targets)
    if (plan['num_clients'],plan['num_classes'],plan['seed'])!=(
            cfg.num_clients,bank.num_classes,cfg.seed):
        raise ValueError('Plan/config/dataset dimensions disagree')
    if plan['allocation_audit'].get('epoch_slicing',{}).get('segments',1)!=cfg.joint_epoch_segments:
        raise ValueError('Plan/config epoch slicing mismatch')
    if plan['allocation_audit'].get('control_kind','federated')!=cfg.joint_control_kind:
        raise ValueError('Plan/config control identity mismatch')
    if plan['allocation_audit']['raw_sample_overlap']!=cfg.sample_overlap_fraction:
        raise ValueError('Plan/config raw sample overlap mismatch')
    if plan['intervals'][-1]['terminal_flush']!=cfg.stream_terminal_flush:
        raise ValueError('Plan/config terminal flush mismatch')
    if cfg.dataset!='synthetic' and not cfg.joint_allow_partial_coverage and len(plan['overlap_audit']['class_owners'])!=bank.num_classes:
        raise ValueError('Production plans must expose every class')
    lookup={x['occurrence_id']:x for x in plan['tasks']}
    holders=defaultdict(set);seen=set();arrivals=[];stages=[]
    for r in plan['intervals']:
        stage=[];counts=defaultdict(int)
        for oid in r['occurrences']:
            x=lookup[oid];cid=x['client_id'];classes=tuple(x['classes'])
            counts[cid]+=1
            if counts[cid]>cfg.local_tasks_per_round:
                raise ValueError('Configured local task bound is below planned depth')
            arrivals.append(Arrival(r['interval_id'],cid,r['interval_id'],
                                    tuple(x['train_indices']),classes,
                                    bool(set(classes)&seen),(),x['head_id']))
            for label in classes:
                holders[label].add(cid)
                if label not in seen:
                    stage.append(label);seen.add(label)
        stages.append(stage)
    return JointScenario(arrivals,stages,dict(holders),cfg.num_clients,bank.num_classes,
                         tuple(plan['allocation_audit'].get('duplicate_sample_ids',())),
                         tuple(plan['allocation_audit']['unused_training_indices']),plan)
