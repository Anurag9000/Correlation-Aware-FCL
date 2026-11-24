"""Simultaneous current-data access oracle preserving original task heads."""
from collections import Counter


def pool_events(events):
    sources=sorted(events,key=lambda e:e['atomic_step'])
    if not sources:raise ValueError('Empty pool')
    indices=[int(i) for e in sources for i in e['indices']]
    if len(set(indices))!=len(indices):
        raise ValueError('Pooled oracle requires unique fresh sample IDs')
    return {**sources[0],'indices':indices,
            'class_ids':tuple(sorted({c for e in sources for c in e['class_ids']})),
            'source_events':sources,'padding_indices':[],
            'current_head_by_index':{int(i):e['task_id'] for e in sources for i in e['indices']}}


def deliver_pool(client,raw,round_id):
    """Expose every original task at the beginning of the pooled intervention."""
    for event in raw['source_events']:
        labels=tuple(event['class_ids'])
        client.arrival(event['indices'],labels,event['global_stage'])
        client.local_experience+=1
        for idx in event['indices']:
            client.sample_first_local_experience.setdefault(int(idx),client.local_experience)
        for c in labels:
            client.first_seen_local_experience.setdefault(int(c),client.local_experience)
            if event['is_recurrence']:client.recurrence_rounds[int(c)]=round_id
        client.local_training_events.append({
            'round':round_id,'local_experience':client.local_experience,
            'class':None,'classes':list(labels),'task_id':event['task_id'],
            'global_stage':event['global_stage'],'recurrence':event['is_recurrence'],
            'selected_for_upload':False,'atomic_step':event['atomic_step'],
            'occurrence_id':event['occurrence_id'],'semantic_task_id':event['semantic_task_id'],
            'simultaneous_access':True})
    client.current_indices=list(raw['indices'])


def occurrence_details(raw,result,cid,experience):
    """Keep occurrence identity without pretending the group trained sequentially.

    Optimizer and replay work is reported once for the group; per-occurrence
    metrics all refer to the same before/after pooled model.
    """
    sources=raw['source_events'];rows=[]
    group_id=f"pool_client_{cid}_atomic_{raw['atomic_step']}"
    group={k:v for k,v in result.items() if k not in ('weights','local_validation_pre_task','local_validation_post_task')}
    for j,event in enumerate(sources):
        rows.append({'client':cid,'atomic_step':event['atomic_step'],
            'occurrence_id':event['occurrence_id'],'semantic_task_id':event['semantic_task_id'],
            'local_experience':experience-len(sources)+j+1,'task_ordinal':j+1,
            'is_selected':False,'simultaneous_access':True,'pooled_group_id':group_id,
            'examples':len(event['indices']),'steps':None,
            'replay_sampled_examples':None,
            'pooled_group_training':group if j==0 else None,
            'local_validation_pre_task':result.get('local_validation_pre_task'),
            'local_validation_post_task':result.get('local_validation_post_task')})
    return rows
