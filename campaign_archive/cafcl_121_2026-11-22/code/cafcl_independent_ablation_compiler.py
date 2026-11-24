#!/usr/bin/env python3
"""Deterministic schedule compiler and mathematical preflight for CAFCL A-G."""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

SEED=0
NCLASS=100
BASE={"dataset":"cifar100","method":"replay","aggregator":"fedavg","learning_setting":"task_il",
      "num_clients":5,"classes_per_task":2,"epochs_per_task":10,"batch_size":64,
      "lr":0.03,"momentum":0.9,"weight_decay":0.0005,"optimizer_state_policy":"persist",
      "training_augmentation":"cifar_crop_flip","replay_ratio":1.0,"memory_per_class":50,
      "replay_reservoir_fraction":1/9,"replay_update_policy":"sampled_mixed",
      "participation":"full","sync_boundary":"after_complete_client_task",
      "terminal_flush":"communication_only","seed":SEED}

def item(r,c,labels,kind='fresh',source=None,group=None):
    return {"sync":r,"client":c,"task_instance":f"r{r:02d}_c{c:02d}","classes":list(labels),
            "recurrence_kind":kind,"source_task":source,"overlap_group":group}

def fresh_pair(ptr): return (ptr,ptr+1)

def compile_a(n, setting):
    tasks=[]; ptr=0; counts=[NCLASS//(2*n)]*n
    if n==3: counts=[17,17,16]
    else: counts=[25 if n==2 else 5]*n
    for r in range(max(counts)):
        for c in range(n):
            if r>=counts[c]: continue
            pair=fresh_pair(ptr);ptr+=2;tasks.append(item(r,c,pair))
    cfg={**BASE,"num_clients":n,"learning_setting":setting,"classes_per_task":2,
         "tasks_per_client":counts,"learning_syncs":max(counts),"class_il_head_width":100 if setting=='class_il' else None}
    return cfg,tasks

def compile_b(delay):
    # One canonical 18-sync schedule is shared by all delay arms. The target
    # pair returns in slot d+1 (zero-based) after exactly d intervening syncs;
    # swap it with that slot's task and move the displaced task to sync index10.
    # Thus all arms retain the same task multiset and per-class example counts.
    ptr=0;owned=defaultdict(list);tasks=[]
    for r in range(10):
        for c in range(5):
            pair=fresh_pair(ptr);ptr+=2;owned[c].append(pair);tasks.append(item(r,c,pair))
    target=(0,1)
    for r in range(10,18):
        for c in range(5):
            if c==0 and r==10:
                pair=target;kind='reserved_target_return';source='r00_c00'
            elif c==0:
                choices=owned[c][1:]  # keep target pair absent before its scheduled return
                pair=choices[(r+c)%len(choices)];kind='filler_recurrence';source=f'owner_{c}'
            else:
                choices=owned[c]
                pair=choices[(r+c)%len(choices)];kind='filler_recurrence';source=f'owner_{c}'
            tasks.append(item(r,c,pair,kind,source))
    event=delay+1
    target_slot=next(t for t in tasks if t['client']==0 and t['sync']==event)
    reserved_slot=next(t for t in tasks if t['client']==0 and t['sync']==10)
    displaced=tuple(target_slot['classes'])
    target_slot['classes']=list(target)
    target_slot['recurrence_kind']='temporal_return'
    target_slot['source_task']='r00_c00'
    prior=next((t for t in tasks if t['client']==0 and t['sync']<10 and tuple(t['classes'])==displaced),None)
    reserved_slot['classes']=list(displaced)
    reserved_slot['recurrence_kind']='displaced_task_slot'
    reserved_slot['source_task']=prior['task_instance'] if prior else None
    for t in tasks:
        if t['client']==0 and t['sync']==10:
            t['displaced_from_sync']=event
            t['displaced_from_task']=target_slot['task_instance']
        if t['client']==0 and t['sync']==event:
            t['swap_with_sync']=10
    cfg={**BASE,"family":"B","recurrence_gap_syncs":delay,"learning_syncs":18,
         "recurrence_return_sync":event+1,"target_task_classes":list(target),
         "matched_task_multiset":True,"swap_sync_zero_based":10,
         "displaced_task_classes":list(displaced),
         "schedule_rule":"swap target recurrence into requested slot; move displaced task to learning sync 11"}
    return cfg,tasks

def compile_c(duration):
    rng=random.Random(SEED+200);ptr=0;owned=defaultdict(list);tasks=[]
    hidden_order=__import__("numpy").random.default_rng(SEED+771721).permutation(5).tolist()
    for r in range(16):
        order=list(range(5));rng.shuffle(order)
        for c in order:
            if r<10:
                pair=fresh_pair(ptr);ptr+=2;owned[c].append(pair);tasks.append(item(r,c,pair));
            else:
                pair=owned[c][(r+c)%len(owned[c])]
                tasks.append(item(r,c,pair,'post_coverage_recurrence',f"owner_{c}"))
    cohorts=[]
    for r in range(16):
        hidden=hidden_order[(r//duration)%5]
        cohorts.append({"sync":r,"uploaded_clients":[c for c in range(5) if c!=hidden],"hidden_clients":[hidden]})
    cfg={**BASE,"participation":"4/5 with seeded rotating outage cohorts","family":"C","outage_syncs":duration,"offline_duration_rounds":duration,"selector":"random","learning_syncs":16,
         "participation_schedule":cohorts,"post_coverage_filler":"fixed 6-sync within-owner recurrence tail"}
    return cfg,tasks

def compile_d(k):
    rng=random.Random(SEED+300);ptr=0;tasks=[];prev={}
    for c in range(5):
        pair=fresh_pair(ptr);ptr+=2;prev[c]=item(0,c,pair);tasks.append(prev[c])
    r=1
    while ptr<NCLASS:
        srcs=rng.sample(range(5),k)
        perms=list(__import__('itertools').permutations(range(5),k))
        rng.shuffle(perms)
        dests=next((p for p in perms if all(s!=d for s,d in zip(srcs,p))),None)
        if dests is None: raise RuntimeError('No cross-client task permutation')
        row={};used=set()
        for src,dst in zip(srcs,dests):
            row[dst]=item(r,dst,prev[src]['classes'],'cross_step_carryover',prev[src]['task_instance'])
            row[dst]['carryover']=True;used.add(dst)
        for c in range(5):
            if c in used:continue
            if ptr+1<NCLASS:
                pair=fresh_pair(ptr);ptr+=2;row[c]=item(r,c,pair)
            else:
                candidates=[prev[c]]
                row[c]=item(r,c,candidates[0]['classes'],'tail_recurrence',candidates[0]['task_instance'])
        tasks.extend(row[c] for c in range(5));prev=row;r+=1
        if r>60:raise RuntimeError('Unexpected D schedule length')
    cfg={**BASE,"family":"D","cross_step_shared_clients":k,"learning_syncs":r,
         "unique_classes_first_seen":NCLASS}
    return cfg,tasks

def compile_e(k):
    rng=random.Random(SEED+400);ptr=0;tasks=[];hist=defaultdict(list);r=0
    while ptr<NCLASS:
        clients=list(range(5));rng.shuffle(clients);shared=clients[:k]
        if ptr+1<NCLASS: shared_pair=fresh_pair(ptr);ptr+=2;shared_kind='fresh_shared'
        else: shared_pair=(0,1);shared_kind='tail_shared'
        group=f"shared_r{r}"
        for c in shared: 
            t=item(r,c,shared_pair,'within_step_shared',None,group);tasks.append(t);hist[c].append(t)
        for c in clients[k:]:
            if ptr+1<NCLASS: pair=fresh_pair(ptr);ptr+=2;kind='fresh'
            else:
                prior=hist[c] or [t for t in tasks if t['client']==c]
                pair=tuple(prior[(r+c)%len(prior)]['classes']);kind='tail_recurrence'
            t=item(r,c,pair,kind,None);tasks.append(t);hist[c].append(t)
        r+=1
    cfg={**BASE,"family":"E","within_step_shared_clients":k,"learning_syncs":r,
         "shared_task_class_width":2}
    return cfg,tasks

def compile_f(width):
    tasks=[]
    if width==3:
        for r in range(7):
            w=3 if r<6 else 2
            for c in range(5):
                start=(r*5+c)*3 if r<6 else 90+c*2
                pair=tuple(range(start,start+w));tasks.append(item(r,c,pair,'remainder_width_2' if w==2 else 'fresh'))
        rounds=7;counts=[7]*5
    else:
        rounds=100//(5*width)
        for r in range(rounds):
            for c in range(5):
                start=(r*5+c)*width;labels=tuple(range(start,start+width));tasks.append(item(r,c,labels))
        counts=[rounds]*5
    cfg={**BASE,"family":"F","classes_per_task_target":width,"learning_syncs":rounds,
         "tasks_per_client":counts,"task_widths":sorted(set(len(t['classes']) for t in tasks))}
    return cfg,tasks

def compile_g(k):
    schedule_rng_seed=SEED+500
    rng=random.Random(schedule_rng_seed);order=list(range(100));rng.shuffle(order);ptr=0;seen=set();tasks=[];r=0
    while len(seen)<100:
        clients=list(range(5));rng.shuffle(clients);prev=None
        for j,c in enumerate(clients):
            labels=[]
            if j==0:
                nnew=min(5,100-ptr);labels.extend(order[ptr:ptr+nnew]);ptr+=nnew
                candidates=[x for x in order[:ptr] if x not in labels]
                while len(labels)<5:
                    x=next(x for x in candidates if x not in labels);labels.append(x)
            else:
                inherited=rng.sample(list(prev),k);labels.extend(inherited)
                needed=5-k;nnew=min(needed,100-ptr);labels.extend(order[ptr:ptr+nnew]);ptr+=nnew
                candidates=[x for x in order[:ptr] if x not in labels]
                while len(labels)<5:
                    # Keep exact intersection with the immediately prior task.
                    x=next(x for x in candidates if x not in labels and x not in prev);labels.append(x)
            inter=len(set(labels)&set(prev)) if prev is not None else None
            t=item(r,c,tuple(labels),'within_step_class_overlap',None,f"overlap_{k}")
            t['ordered_client_slot']=j;t['intersection_with_previous']=inter;tasks.append(t)
            seen.update(labels);prev=tuple(labels)
        r+=1
        if r>20: raise RuntimeError('Unexpected G schedule length')
    cfg={**BASE,"family":"G","classes_per_task_target":5,"adjacent_shared_classes":k,
         "learning_syncs":r,"schedule_rng_seed":schedule_rng_seed}
    return cfg,tasks

def specs():
    for n in (2,3,10):
        for setting in ('task_il','class_il'):
            cfg,tasks=compile_a(n,setting);yield 'A',f'clients_{n}_{setting}',cfg,tasks
    for d in (1,2,5,10,15):
        cfg,tasks=compile_b(d);yield 'B',f'recur_gap_{d}',cfg,tasks
    for d in (1,2,3,5,10,15):
        cfg,tasks=compile_c(d);yield 'C',f'hidden_{d}',cfg,tasks
    for k in (2,3,4):
        cfg,tasks=compile_d(k);yield 'D',f'cross_step_{k}',cfg,tasks
    for k in (2,3,4):
        cfg,tasks=compile_e(k);yield 'E',f'within_step_{k}',cfg,tasks
    for w in (3,5,10):
        cfg,tasks=compile_f(w);yield 'F',f'task_width_{w}',cfg,tasks
    for k in (2,3,4):
        cfg,tasks=compile_g(k);yield 'G',f'width5_overlap_{k}',cfg,tasks

def validate(family,name,cfg,tasks):
    if not tasks: raise ValueError(f'{name}: no task assignments')
    clients=cfg['num_clients']; grouped=defaultdict(list); bysync=defaultdict(dict)
    for t in tasks:
        labels=t['classes'];c=t['client'];r=t['sync']
        if len(labels)!=len(set(labels)) or any(not 0<=x<100 for x in labels): raise ValueError(f'{name}: bad labels {t}')
        grouped[c].append(t);bysync[r][c]=t
    union={x for t in tasks for x in t['classes']}
    if union!=set(range(100)): raise ValueError(f'{name}: coverage missing={sorted(set(range(100))-union)}')
    if family=='A' and cfg['num_clients']==3:
        if sorted(len(grouped[c]) for c in range(3))!=[16,17,17]: raise ValueError(f'{name}: invalid 3-client tail')
    if family=='B':
        first=next(t for t in tasks if t['client']==0 and tuple(t['classes'])==(0,1))
        ret=next(t for t in tasks if t['recurrence_kind']=='temporal_return')
        if ret['sync']-first['sync']-1!=cfg['recurrence_gap_syncs']: raise ValueError(f'{name}: recurrence gap off by one')
        target_count=sum(t['client']==0 and tuple(t['classes'])==(0,1) for t in tasks)
        displaced=next(t for t in tasks if t['client']==0 and t['sync']==10)
        if target_count!=2 or displaced['recurrence_kind']!='displaced_task_slot': raise ValueError(f'{name}: recurrence swap invalid')
    if family=='C':
        d=cfg['outage_syncs'];order=__import__("numpy").random.default_rng(SEED+771721).permutation(5).tolist()
        for cset in cfg['participation_schedule']:
            should_hide=order[(cset['sync']//d)%5]
            if cset['hidden_clients']!=[should_hide]: raise ValueError(f'{name}: outage schedule error')
    if family=='D':
        for r in range(1,cfg['learning_syncs']):
            carried=sum(t.get('carryover',False) for t in bysync[r].values())
            if carried!=cfg['cross_step_shared_clients']: raise ValueError(f'{name}: expected {cfg["cross_step_shared_clients"]} carryovers at sync {r}, got {carried}')
    if family=='E':
        k=cfg['within_step_shared_clients']
        for r in range(cfg['learning_syncs']):
            tagged=[t for t in tasks if t['sync']==r and t.get('overlap_group')==f'shared_r{r}']
            if len(tagged)!=k: raise ValueError(f'{name}: shared group size mismatch at {r}')
            if len({tuple(t['classes']) for t in tagged})!=1: raise ValueError(f'{name}: shared class sets differ')
    if family=='F':
        for c in range(5):
            if len(grouped[c])!=cfg['learning_syncs']: raise ValueError(f'{name}: client task count mismatch')
    if family=='G':
        k=cfg['adjacent_shared_classes']
        for r in range(cfg['learning_syncs']):
            row=sorted([t for t in tasks if t['sync']==r],key=lambda x:x['ordered_client_slot'])
            if len(row)!=5 or any(len(t['classes'])!=5 for t in row): raise ValueError(f'{name}: width/row mismatch')
            for j in range(1,5):
                if len(set(row[j-1]['classes'])&set(row[j]['classes']))!=k: raise ValueError(f'{name}: adjacent overlap mismatch')
    return {"run_id":f'{family}_{name}_seed0',"family":family,"schedule_validated":True,
            "learning_syncs":max(t['sync'] for t in tasks)+1,"task_instances":len(tasks),
            "classes_covered":100,"tasks_per_client":{str(c):len(grouped[c]) for c in range(clients)},
            "task_width_histogram":dict(Counter(map(lambda t:len(t['classes']),tasks))),
            "terminal_flush":"communication_only","seed":SEED}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);rows=[];b_multiset=None;b_occurrences=None
    for fam,name,cfg,tasks in specs():
        summary=validate(fam,name,cfg,tasks)
        if fam=='B':
            from collections import Counter
            multiset=Counter((t['client'],tuple(sorted(t['classes']))) for t in tasks)
            occurrences=Counter(x for t in tasks for x in t['classes'])
            if b_multiset is None: b_multiset,b_occurrences=multiset,occurrences
            elif multiset!=b_multiset or occurrences!=b_occurrences: raise ValueError('B delay arms must preserve task multiset and class occurrence counts')
        cfg['family']=fam;cfg['run_id']=summary['run_id'];summary.update(cfg)
        dest=root/fam/f'{name}_seed0';dest.mkdir(parents=True,exist_ok=False)
        (dest/'config.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
        (dest/'schedule.json').write_text(json.dumps(tasks,indent=2,sort_keys=True)+'\n')
        (dest/'plan_validation.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        rows.append(summary)
    manifest={"campaign":"cafcl_independent_cifar100_ablations_A_G","seed":SEED,
              "run_count":len(rows),"families":dict(Counter(x['family'] for x in rows)),
              "runs":rows,"note":"schedule compilation is not a training result"}
    (root/'campaign_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"run_count":len(rows),"families":manifest['families'],"output":str(root)},indent=2))
if __name__=='__main__': main()
