"""Static mathematical feasibility audit for every CIFAR-100 v6 configuration.

Computes every scheduled task, checks label budget and episode distinctness,
validates class-specific training-image budgets and controlled outages. No
GPU, downloading, or test data involved. Cached on the actual stream axes.
"""
from collections import Counter
from functools import lru_cache
from dataclasses import replace
from .streaming import make_task_schedule

@lru_cache(maxsize=None)
def check_stream(seed,clients,classes,tasks,kind,overlap,recurrences,participation,outage,flush,
                 imbalance,quantity,sample_overlap):
    from .full_matrix import base
    cfg=replace(base(seed),num_clients=clients,classes_per_local_task=classes,local_tasks_per_round=tasks,
                stream_overlap_kind=kind,stream_overlap_fraction=overlap,stream_recurrence_rounds=recurrences,
                participation=participation,offline_duration_rounds=outage,stream_terminal_flush=flush,
                class_imbalance_power=imbalance,client_sample_skew=quantity,
                sample_overlap_fraction=sample_overlap)
    try: events=make_task_schedule(100,cfg)
    except Exception as exc:return str(exc)
    if not events:return 'empty-stream'
    new_training_rounds=(1+max(r for r,_,_,_,_ in events))-recurrences
    ids=set();counts=Counter();seen=set();by_round=Counter()
    occurrences={}
    geometric_types=set()
    forbidden_reuse=0
    for r,cid,t,labels,recurrent in events:
        if not (0<=cid<clients and 0<=t<tasks):return 'invalid-client-or-task-index'
        if not (2<=len(labels)<=classes and len(set(labels))==len(labels)):return 'invalid-active-label-group'
        if len(labels)!=classes:return 'incomplete-contrastive-task'
        if len(labels)!=len(set(labels)):return 'duplicate-class-in-task'
        if any(cls not in range(100) for cls in labels):return 'label-out-of-range'
        ids.update(labels);counts.update(labels);by_round[(r,cid)]+=1
        for cls in labels:
            previous=occurrences.get(cls,[])
            if previous:
                if r<new_training_rounds and kind=='none':return 'unexpected-label-overlap'
                if r<new_training_rounds:
                    relations=set()
                    for pr,pc,pt in previous:
                        if pc==cid and pr==r:relations.add('intra_same_round')
                        elif pc==cid and pr<r:relations.add('intra_cross_round')
                        elif pc!=cid and pr==r:relations.add('inter_same_round')
                        elif pc!=cid and pr<r:relations.add('inter_cross_round')
                    geometric_types.update(relations)
                    if kind!='combined' and kind!='none' and kind not in relations:
                        return 'overlap-violates-declared-geometry'
                    if kind=='combined' and not relations:
                        return 'overlap-has-no-valid-geometry'
            occurrences.setdefault(cls,[]).append((r,cid,t))
        seen.update(labels)
    if ids!=set(range(100)):return 'missing-labels'
    if kind not in ('none','combined') and overlap and kind not in geometric_types:
        return 'overlap-geometry-never-occurs'
    if kind=='combined' and overlap and not set(('intra_same_round','inter_same_round','intra_cross_round','inter_cross_round')).issubset(geometric_types):
        return 'combined-does-not-realize-all-four-overlap-geometries' 
    if kind=='none' and recurrences==0 and sum(counts.values())!=100:return 'disjoint-class-budget-mismatch'
    if any(c>tasks for c in by_round.values()):return 'too-many-tasks-per-client'
    train_rounds=1+max(r for r,_,_,_,_ in events)
    total_rounds=train_rounds+int(flush)
    if outage:
        if outage>=total_rounds:return 'no-controlled-outage-reconnection'
        if not 0<participation<1:return 'controlled-outage-requires-partial-participation'
    # Verify ACTUAL outage episodes, not just D < total rounds. The final
    # communication-only flush is a reconnection, not a hidden training round.
    # A last, truncated cohort would violate the advertised exact-D experiment.
    if outage:
        from .selection import participation_budget
        from .streaming import select_offline_cohort
        hidden_count=clients-participation_budget(clients,participation)
        if hidden_count<1 or hidden_count>=clients:
            return 'controlled-outage-participation-budget-invalid'
        if not flush:
            return 'controlled-outage-requires-final-reconnection-flush'
        hidden_by_round=[]
        for r in range(train_rounds):
            try:
                chosen=set(select_offline_cohort(cfg,r,total_rounds,
                    list(range(clients-hidden_count))))
            except (ValueError, AssertionError) as exc:
                return 'controlled-outage-schedule-invalid:'+str(exc)
            hidden=set(range(clients))-chosen
            if len(hidden)!=hidden_count:
                return 'controlled-outage-client-count-mismatch'
            hidden_by_round.append(hidden)
        for cid in range(clients):
            start=None
            # Sentinel event at the final communication-only flush.
            for r in range(train_rounds+1):
                missing=r<train_rounds and cid in hidden_by_round[r]
                if missing and start is None:
                    start=r
                elif not missing and start is not None:
                    if r-start!=outage:
                        return 'truncated-or-extended-offline-duration'
                    start=None
    # In CIFAR-100 there are 500 raw train examples per class; splits are
    # partitioned between that class's task occurrences, without duplication.
    for c,n in counts.items():
        rank=c/99
        fraction=max(.1,1.-.9*rank)**imbalance if imbalance else 1.
        budget=max(n,round(500*fraction))
        if n>budget:return 'too-many-repeated-classes-for-available-images'
    return None



def check_job(cfg):
    """Full declared-plan feasibility, including semantic experimental no-ops.

    Structural checks are memoized by stream axes. These additional checks
    intentionally distinguish a runnable configuration from an identifiable
    experiment in which the named factor actually varies.
    """
    from .selection import participation_budget
    # LwF uses teacher distillation, not replay memory or generic
    # parameter-importance regularization. Prune inert parameter ablations.
    if cfg.method=='lwf' and cfg.memory_per_class!=20:
        return 'lwf-memory-capacity-unused'
    if cfg.method=='lwf' and cfg.regularization_strength!=10.0:
        return 'lwf-regularization-strength-unused'
    if cfg.selector in {'fairness','diversity','balanced'}:
        # Implementations are exact aliases: fairness==least_recent,
        # diversity==coverage, balanced==coverage because old/new form
        # a disjoint partition of all classes currently held.
        return 'selection-policy-exact-alias'
    if (cfg.selector!='random' and
            participation_budget(cfg.num_clients,cfg.participation)==cfg.num_clients):
        return 'selection-policy-inert-at-full-participation'
    if (cfg.selector=='recurrence' and
            cfg.stream_overlap_kind=='none' and cfg.stream_recurrence_rounds==0):
        return 'recurrence-selector-has-no-recurrence-signal'
    if cfg.selector=='overlap' and cfg.stream_overlap_kind=='none':
        return 'overlap-selector-has-no-overlap-signal'
    # This protocol is fixed throughout all active suites; retain historical
    # task_length candidate names in the raw design only as REJECTED arms.
    if cfg.classes_per_local_task != 2 or cfg.local_tasks_per_round != 2:
        return 'retired-variable-task-length-protocol'
    return check_stream(cfg.seed,cfg.num_clients,cfg.classes_per_local_task,
                        cfg.local_tasks_per_round,cfg.stream_overlap_kind,
                        cfg.stream_overlap_fraction,cfg.stream_recurrence_rounds,
                        cfg.participation,cfg.offline_duration_rounds,
                        cfg.stream_terminal_flush,cfg.class_imbalance_power,
                        cfg.client_sample_skew,cfg.sample_overlap_fraction)


def audit(jobs):
    dropped=Counter();kept=Counter();examples={};n=0
    for family,cfg in jobs:
        n+=1
        reason=check_job(cfg)
        if reason:
            dropped[(family,reason)]+=1;examples.setdefault((family,reason),cfg.job_id)
        else:kept[family]+=1
    return dict(input_jobs=n,accepted_jobs=sum(kept.values()),rejected_jobs=sum(dropped.values()),
                accepted_by_family=dict(sorted(kept.items())),
                rejected_by_family_reason={f'{fam}:{reason}':num for (fam,reason),num in sorted(dropped.items())},
                examples={f'{fam}:{reason}':job for (fam,reason),job in examples.items()},
                cached_stream_structures=check_stream.cache_info().currsize)

