"""CIFAR-100 communication-interval curriculum with canonical class-indexed tasks.

Each client learns ``local_tasks_per_round`` sequential multi-class episodes BEFORE
server selection. Each episode has ``classes_per_local_task`` distinct labels.
No task ID or task-specific classifier is created. This generator is deterministic,
uses only original TRAIN images, and never intentionally duplicates raw images
unless ``sample_overlap_fraction`` is nonzero.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import numpy as np
from .data import Arrival, Scenario

OVERLAP_KINDS = ('none','intra_same_round','intra_cross_round',
                 'inter_same_round','inter_cross_round','combined')


def make_task_schedule(num_classes, cfg):
    """Return ordered (round, client, task ordinal, class IDs, recurrence flag).

    Default 10 clients × 2 tasks × 2 classes gives exactly the example
    0,1 / 20,21 on client 0 in round 0, then 40,41 / 60,61 in round 1.
    No-overlap never reuses a class and never creates a singleton task.
    """
    if cfg.classes_per_local_task < 2 or cfg.local_tasks_per_round < 1:
        raise ValueError('Every local task needs >=2 classes and positive task count')
    rng = np.random.default_rng(cfg.seed + 73019)
    order = list(cfg.class_order_override) if cfg.class_order_override else list(range(num_classes))
    if len(order)!=num_classes or set(order)!=set(range(num_classes)):
        raise ValueError('Stream class order must be a permutation of dataset labels')
    if cfg.stream_overlap_kind not in OVERLAP_KINDS:
        raise ValueError('Unsupported overlap kind')
    if cfg.stream_overlap_kind == 'none' and num_classes % cfg.classes_per_local_task:
        raise ValueError('No-overlap stream cannot end in a singleton/incomplete local task')
    if cfg.stream_overlap_kind=='none' and cfg.stream_overlap_fraction:
        raise ValueError('No-overlap stream cannot request label reuse')
    if cfg.sample_overlap_fraction and cfg.stream_overlap_kind=='none':
        raise ValueError('Raw sample overlap requires repeated class ownership')
    K=cfg.num_clients; C=cfg.classes_per_local_task; T=cfg.local_tasks_per_round
    current=0; rnd=0; history={cid:[] for cid in range(K)}
    schedule=[]; seen=set()
    # Reuse only *previously assigned* classes, including earlier tasks within
    # this interval; never leak a future label into a previous local task.
    while current<len(order):
        made=0
        current_round={cid:[] for cid in range(K)}
        for task in range(T):
            for cid in range(K):
                if current>=len(order):break
                eligible=[]
                kind=cfg.stream_overlap_kind
                if kind=='combined':
                    kind=('intra_same_round','inter_same_round',
                          'intra_cross_round','inter_cross_round')[(rnd+task+cid)%4]
                if kind=='intra_same_round' and current_round[cid]:
                    eligible=list(current_round[cid][-1])
                elif kind=='inter_same_round' and cid>0 and current_round[cid-1]:
                    eligible=list(current_round[cid-1][-1])
                elif kind=='intra_cross_round':
                    eligible=next((list(v) for r,v in reversed(history[cid]) if r<rnd),[])
                elif kind=='inter_cross_round':
                    eligible=next((list(v) for r,v in reversed(history[(cid+1)%K]) if r<rnd),[])
                overlap=bool(eligible and cfg.stream_overlap_fraction>0 and
                             rng.random()<cfg.stream_overlap_fraction)
                available=len(order)-current
                take=C-int(overlap)
                if available<take:
                    if cfg.stream_overlap_kind=='none':
                        raise ValueError('Insufficient distinct classes to make final task')
                    # Finish a partial last event with already observed labels,
                    # while retaining >=2 distinct labels for meaningful CE.
                    take=available
                fresh=order[current:current+take]
                current+=take
                labels=list(fresh)
                if overlap:
                    reuse=next((v for v in eligible if v not in labels),None)
                    if reuse is not None:labels.insert(0,reuse)
                if len(labels)<C:
                    candidates=[v for v in reversed(sorted(seen)) if v not in labels]
                    labels+=candidates[:C-len(labels)]
                if len(labels)<2 or len(set(labels))!=len(labels):
                    raise ValueError('Not enough distinct labels for contrastive episode')
                recurrence=any(cls in seen for cls in labels)
                schedule.append((rnd,cid,task,tuple(labels),recurrence))
                seen.update(labels)
                current_round[cid].append(tuple(labels));history[cid].append((rnd,tuple(labels)))
                made+=1
            if current>=len(order):break
        if not made:raise RuntimeError('Stream scheduler made no progress')
        rnd+=1
        if rnd>num_classes*4:raise RuntimeError('Non-terminating stream schedule')
    # Optional *explicit* recurrent rounds to study lengthy disconnections.
    # These use fresh TRAIN examples from previously encountered labels, not
    # new labels and not repeated raw images (unless intentionally configured).
    for extra in range(cfg.stream_recurrence_rounds):
        for task in range(T):
            for cid in range(K):
                own=[v for _,group in history[cid] for v in group]
                if not own:continue
                unique=list(dict.fromkeys(own))
                if len(unique)<C:continue
                start=(extra*T+task)*C % len(unique)
                labels=tuple(unique[(start+j)%len(unique)] for j in range(C))
                if len(set(labels))<2:continue
                schedule.append((rnd,cid,task,labels,True))
        rnd+=1
    return schedule


def compile_stream_scenario(bank,cfg):
    if cfg.dataset not in ('cifar100','synthetic'):
        raise ValueError('Streaming research curriculum is supported on CIFAR-100 only')
    if cfg.relation!='natural' and not cfg.class_order_override:
        from dataclasses import replace
        from .data import class_order
        cfg=replace(cfg,class_order_override=tuple(class_order(bank,cfg.relation)))
    schedule=make_task_schedule(bank.num_classes,cfg)
    # Partition each label's raw TRAIN images among occurrences of that label,
    # so class overlap is not secretly raw-example overlap.
    rng=np.random.default_rng(cfg.seed+9981)
    pool=defaultdict(list)
    for index,label in enumerate(bank.train.targets):pool[int(label)].append(int(index))
    counts=Counter(cls for _,_,_,labels,_ in schedule for cls in labels)
    for cls,indices in pool.items():
        rng.shuffle(indices)
        # Quantity and long-tail interventions consume only training examples.
        fraction=cfg.class_sample_fraction.get(str(cls),1.)
        if cfg.class_imbalance_power:
            rank=cls/max(1,bank.num_classes-1)
            fraction*=max(.1,1.-.9*rank)**cfg.class_imbalance_power
        limit=max(counts[cls],int(round(len(indices)*fraction)))
        pool[cls]=indices[:min(len(indices),limit)]
    parts={}; cursors={cls:0 for cls in pool};occ=Counter()
    for _,_,_,labels,_ in schedule:
        for cls in labels:
            occ[cls]+=1
    used=Counter(); dup=[]; arrivals=[];label_holders=defaultdict(set)
    for rnd,cid,task,labels,recur in schedule:
        ids=[]
        for cls in labels:
            index=used[cls];used[cls]+=1
            items=pool[cls]
            left=math.floor(index*len(items)/counts[cls]);right=math.floor((index+1)*len(items)/counts[cls])
            sample=items[left:right]
            if cfg.sample_overlap_fraction and index>0:
                n=int(round(len(sample)*cfg.sample_overlap_fraction))
                reused=items[:min(n,left)]
                sample=sample+reused
                dup.extend(reused)
            if cfg.client_sample_skew:
                # Client-level quantity skew without reassigning class labels;
                # dropped examples are accounted for as unused training data.
                weight=1./(1.+cfg.client_sample_skew*cid/max(1,cfg.num_clients-1))
                sample=sample[:max(1,int(round(len(sample)*weight)))]
            if not sample:raise ValueError(f'Not enough examples for label {cls} across its overlapping tasks')
            ids.extend(sample)
            label_holders[cls].add(cid)
        arrivals.append(Arrival(rnd,cid,rnd,tuple(ids),labels,bool(recur)))
    stages=[[] for _ in range(1+max(r for r,_,_,_,_ in schedule))]
    if cfg.stream_terminal_flush:
        stages.append([])  # communication-only flush to integrate last offline clients
    assigned=set()
    for a in arrivals:
        for label in a.labels:
            if label not in assigned:
                stages[a.round_id].append(label);assigned.add(label)
    all_ids=set(i for a in arrivals for i in a.indices)
    unused=tuple(sorted(set(range(len(bank.train)))-all_ids))
    if set(label_holders)!=set(range(bank.num_classes)):
        raise RuntimeError('Unexposed classes in streaming curriculum')
    return Scenario(arrivals,stages,dict(label_holders),cfg.num_clients,bank.num_classes,
                    tuple(sorted(set(dup))),unused)


def select_offline_cohort(cfg,round_id,total_rounds,standard_selected):
    """Exact-length controlled outages, deterministic per seeded run.

    When offline_duration_rounds>0, a cohort stays offline for exactly D
    consecutive communication rounds and then a different cohort is offline.
    This is a deliberate scheduling intervention superseding the selector.
    """
    if not cfg.offline_duration_rounds:return standard_selected
    if cfg.selector!='random':
        raise ValueError('Controlled outage cohort currently requires random selector')
    n=cfg.num_clients
    absent=n-int(math.floor(cfg.participation*n+0.5))
    if absent<1 or absent>=n:
        raise ValueError('Controlled outages require at least one online and one offline client')
    d=cfg.offline_duration_rounds
    if d>=total_rounds:
        raise ValueError('Outage duration must allow at least one reconnection within the stream')
    rng=np.random.default_rng(cfg.seed+771721)
    order=rng.permutation(n).tolist()
    phase=round_id//d
    start=(phase*absent)%n
    offline={order[(start+j)%n] for j in range(absent)}
    return sorted(set(range(n))-offline)
