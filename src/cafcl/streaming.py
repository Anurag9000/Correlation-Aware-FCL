"""CIFAR-100 communication-interval curricula with class-indexed tasks.

The canonical five-client chain schedule gives every client one adjacent
two-class task per communication interval. It carries the last label seen by
the previous client into the next pair, then carries client five's last label
into client one at the following interval. It partitions TRAIN images by
occurrence and never duplicates raw images unless explicitly configured.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import numpy as np
from .data import Arrival, Scenario
from .task_ids import task_instance_id

OVERLAP_KINDS = ('none','intra_same_round','intra_cross_round',
                 'inter_same_round','inter_cross_round','combined')


def make_task_schedule(num_classes, cfg):
    """Return ordered (round, client, task ordinal, class IDs, recurrence flag).

    With ``stream_client_chain``, each round assigns one adjacent pair to
    every client, extending a single ordered chain across clients and rounds.
    Otherwise the general overlap/task-length research scheduler is used.
    """
    if cfg.stream_client_chain:
        if cfg.classes_per_local_task != 2 or cfg.local_tasks_per_round != 1:
            raise ValueError('Client-chain streaming requires one two-class task per client per sync')
        order=list(cfg.class_order_override) if cfg.class_order_override else list(range(num_classes))
        if len(order)!=num_classes or set(order)!=set(range(num_classes)):
            raise ValueError('Stream class order must be a permutation of dataset labels')
        if num_classes<2:
            raise ValueError('Client-chain stream needs at least two classes')
        schedule=[];seen=set();current=0;round_id=0;previous_pair=None
        while current<num_classes:
            round_pairs=[]
            for cid in range(cfg.num_clients):
                if previous_pair is None:
                    pair=tuple(order[:2]);current=2
                elif current<num_classes:
                    pair=(previous_pair[-1],order[current]);current+=1
                else:
                    # Keep every client active in the final interval. The
                    # finite 100-label/200-slot plan has one unavoidable
                    # third owner (class 98) so class 99 also has two owners.
                    pair=previous_pair
                recurrent=any(label in seen for label in pair)
                schedule.append((round_id,cid,0,pair,recurrent))
                seen.update(pair);round_pairs.append(pair);previous_pair=pair
            round_id+=1
        if cfg.stream_recurrence_rounds:
            history={cid:[] for cid in range(cfg.num_clients)}
            for r,cid,_,labels,_ in schedule:
                history[cid].append((r,labels))
            for extra in range(cfg.stream_recurrence_rounds):
                for cid in range(cfg.num_clients):
                    own=list(dict.fromkeys(v for _,group in history[cid] for v in group))
                    start=(extra*2)%len(own)
                    labels=(own[start],own[(start+1)%len(own)])
                    schedule.append((round_id,cid,0,labels,True))
                round_id+=1
        return schedule
    if (cfg.classes_per_local_task == 2 and cfg.local_tasks_per_round == 1
            and cfg.stream_overlap_kind == 'none'):
        # One fresh, disjoint two-class task per client at each sync. This
        # preserves the V25 local-task/synchronization cadence while removing
        # its cross-client and cross-round class overlap. For CIFAR-100 with
        # five clients this yields ten learning syncs, then the configured
        # communication-only flush.
        if cfg.stream_recurrence_rounds:
            raise ValueError('One-task disjoint streaming cannot also add recurrence rounds')
        if cfg.stream_overlap_fraction or cfg.sample_overlap_fraction:
            raise ValueError('Disjoint one-task streaming cannot request overlap')
        order=list(cfg.class_order_override) if cfg.class_order_override else list(range(num_classes))
        if len(order)!=num_classes or set(order)!=set(range(num_classes)):
            raise ValueError('Stream class order must be a permutation of dataset labels')
        if num_classes % cfg.classes_per_local_task:
            raise ValueError('Disjoint one-task stream cannot end in an incomplete class task')
        schedule=[];current=0;round_id=0
        while current<num_classes:
            for cid in range(cfg.num_clients):
                if current>=num_classes:break
                labels=tuple(order[current:current+cfg.classes_per_local_task])
                if len(labels)!=cfg.classes_per_local_task:
                    raise ValueError('Disjoint one-task stream produced an incomplete class task')
                schedule.append((round_id,cid,0,labels,False))
                current+=cfg.classes_per_local_task
            round_id+=1
        return schedule
    if cfg.classes_per_local_task != 2 or cfg.local_tasks_per_round != 2:
        raise ValueError('Only two initial two-class tasks and one subsequent two-class task are supported')
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
    K=cfg.num_clients; C=2; first_tasks=2
    current=0; rnd=0; history={cid:[] for cid in range(K)}
    schedule=[]; seen=set()
    # Reuse only *previously assigned* classes, including earlier tasks within
    # this interval; never leak a future label into a previous local task.
    while current<len(order):
        made=0
        current_round={cid:[] for cid in range(K)}
        # First interval: four new labels/client. Later: just two/client.
        # Client-first order keeps consecutive task pairs together.
        for cid in range(K):
            for task in range(first_tasks if rnd==0 else 1):
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
                    # The terminal partial event is permitted only at a client
                    # whose declared overlap source can supply all remaining
                    # contrasting labels. Otherwise skip this *client slot*
                    # (do not consume fresh labels) and try the next slot.
                    if available+len(set(eligible))<C:
                        continue
                    take=available
                fresh=order[current:current+take]
                current+=take
                labels=list(fresh)
                if overlap:
                    reuse=next((v for v in eligible if v not in labels),None)
                    if reuse is not None:labels.insert(0,reuse)
                if len(labels)<C:
                    # Tail completion must NOT silently introduce an unrelated
                    # label from another client/round. Use only labels that
                    # satisfy the configured overlap geometry; otherwise this
                    # trial is mathematically infeasible and is pruned.
                    candidates=[v for v in eligible if v not in labels]
                    if len(candidates)<C-len(labels):
                        raise ValueError(f'Incomplete final task cannot be filled within declared overlap geometry (seed={cfg.seed},kind={kind},rnd={rnd},cid={cid},task={task},eligible={eligible},labels={labels},available={available})')
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
        for cid in range(K):
            for task in range(1):
                own=[v for _,group in history[cid] for v in group]
                if not own:continue
                unique=list(dict.fromkeys(own))
                if len(unique)<C:continue
                start=(extra+task)*C % len(unique)
                labels=tuple(unique[(start+j)%len(unique)] for j in range(C))
                if len(set(labels))<2:continue
                schedule.append((rnd,cid,task,labels,True))
        rnd+=1
    return schedule


def compile_stream_scenario(bank,cfg):
    if getattr(cfg,'joint_plan_path',''):
        from .joint.adapter import compile_joint_scenario
        return compile_joint_scenario(bank,cfg)
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
        task_id=task_instance_id(cid,rnd,task,labels)
        arrivals.append(Arrival(rnd,cid,rnd,tuple(ids),labels,bool(recur),(),task_id))
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
    scenario=Scenario(arrivals,stages,dict(label_holders),cfg.num_clients,bank.num_classes,
                      tuple(sorted(set(dup))),unused)
    if cfg.stream_sync_every_batch:
        return expand_stream_to_batch_syncs(scenario,cfg)
    return scenario


def expand_stream_to_batch_syncs(scenario,cfg):
    """Turn each streamed task pass into one FedAvg sync per physical batch."""
    by_round=defaultdict(list)
    for arrival in scenario.arrivals:
        by_round[int(arrival.round_id)].append(arrival)
    expanded=[];stages=[];assigned=set();sync_id=0
    for original_round in sorted(by_round):
        arrivals=sorted(by_round[original_round],key=lambda item:item.client_id)
        if len({a.client_id for a in arrivals})!=len(arrivals):
            raise RuntimeError('Batch-sync expansion expects one task per client per source round')
        for epoch in range(cfg.epochs):
            batches={}
            for arrival in arrivals:
                values=np.asarray(arrival.indices,dtype=np.int64)
                rng=np.random.default_rng(
                    cfg.seed+9_771_203+original_round*1_000_003+
                    arrival.client_id*10_007+epoch*101)
                values=values[rng.permutation(len(values))]
                batches[arrival.client_id]=[
                    tuple(map(int,values[start:start+cfg.batch_size]))
                    for start in range(0,len(values),cfg.batch_size)]
            max_batches=max((len(value) for value in batches.values()),default=0)
            seen_stream_by_client={a.client_id:[] for a in arrivals}
            for batch_index in range(max_batches):
                new_labels=[]
                batch_arrivals=[]
                for arrival in arrivals:
                    client_batches=batches[arrival.client_id]
                    if batch_index>=len(client_batches):
                        continue
                    batch_values=client_batches[batch_index]
                    padding_pool=()
                    if len(batch_values)<cfg.batch_size:
                        padding_pool=tuple(seen_stream_by_client[arrival.client_id])
                        if not padding_pool:
                            padding_pool=tuple(batch_values)
                    batch_arrivals.append(Arrival(
                        sync_id,arrival.client_id,sync_id,
                        batch_values,arrival.labels,arrival.recurrence,padding_pool,
                        arrival.task_id,
                        epoch*len(client_batches)+batch_index,
                        cfg.epochs*len(client_batches),
                        epoch==cfg.epochs-1 and batch_index==len(client_batches)-1))
                    seen_stream_by_client[arrival.client_id].extend(batch_values)
                    for label in arrival.labels:
                        label=int(label)
                        if label not in assigned:
                            assigned.add(label);new_labels.append(label)
                if not batch_arrivals:
                    continue
                expanded.extend(batch_arrivals)
                stages.append(sorted(new_labels))
                sync_id+=1
    if cfg.stream_terminal_flush:
        stages.append([])
    return Scenario(expanded,stages,scenario.label_holders,scenario.num_clients,
                    scenario.num_classes,scenario.duplicate_sample_ids,
                    scenario.unused_sample_ids)


def select_offline_cohort(cfg,round_id,total_rounds,standard_selected):
    """Exact-length controlled outages, deterministic per seeded run.

    When offline_duration_rounds>0, a cohort stays offline for exactly D
    consecutive communication rounds and then a different cohort is offline.
    This is a deliberate scheduling intervention superseding the selector.
    """
    if cfg.stream_offline_policy == 'rotate_once':
        # One client offline per training interval, cyclically covering every
        # client. For five clients/100 classes the new protocol runs NINE
        # training rounds; the old five-round exact-once rule is invalid.
        training_rounds=total_rounds-int(cfg.stream_terminal_flush)
        if training_rounds<cfg.num_clients:
            raise ValueError('rotate_once requires at least one outage per client')
        if cfg.stream_overlap_kind!='none' or cfg.stream_recurrence_rounds:
            raise ValueError('rotate_once pilot requires disjoint, nonrecurrent classes')
        if len(standard_selected)!=cfg.num_clients-1:
            raise ValueError('rotate_once requires exactly one offline client')
        if round_id>=training_rounds:return list(range(cfg.num_clients))
        order=np.random.default_rng(cfg.seed+771721).permutation(cfg.num_clients).tolist()
        return [c for c in range(cfg.num_clients) if c!=order[round_id % cfg.num_clients]]
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
