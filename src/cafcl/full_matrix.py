"""CIFAR-100-only class-indexed, sequential-local-task experiment registry.

No CIFAR-10, CIFAR-100/20, or legacy stage-based experiment may enter this
registry. Its finite crosses preserve all 11 local methods and all 9 server
aggregators, and explicitly test *four* distinct overlap causal geometries.
"""
from __future__ import annotations
import argparse,gzip,json,time
from dataclasses import asdict,replace
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
from .config import RunConfig
from .simulator import run_experiment,atomic_json
from .streaming import OVERLAP_KINDS,make_task_schedule
from .models import BACKBONES,resolve_backbone
from .stream_audit import check_job

METHODS=('sgd','gem','agem','replay','ewc','si','er_ace','der','derpp','mas','lwf')
AGGS=('fedavg','uniform','fedavgm','fedadam','class_balanced','inverse_participation',
      'conflict_aware','conflict_projection','retention_aware')
SEEDS=(0,1,2)


def fingerprint(cfg):
    x=cfg.to_dict();x.pop('job_id',None);x.pop('block_id',None)
    return sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def base(seed=0,model=None):
    return RunConfig(dataset='cifar100',stream_protocol=True,class_pair_protocol='strict',
        learning_setting='class_il',classes_per_local_task=2,local_tasks_per_round=2,
        stream_overlap_kind='none',stream_overlap_fraction=0,
        num_clients=10,participation=.8,selector='random',seed=seed,
        micro_batch_size=16,
        stages=1,rounds_per_stage=1,batch_size=1024,
        model=resolve_backbone(model or 'vit'))


def raw_design(seeds=SEEDS):
    for seed in seeds:
        for method,agg,kind,rate in product(METHODS,AGGS,OVERLAP_KINDS,(1.,.8,.5)):
            yield 'balanced',seed,dict(method=method,aggregator=agg,
                stream_overlap_kind=kind,scenario='disjoint' if kind=='none' else 'overlap',
                stream_overlap_fraction=0 if kind=='none' else .5,participation=rate)
        # Independently ablate the number of contrasted classes and number
        # of *sequential* local tasks processed before a communication round.
        for method,agg,kind,classes,tasks in product(
                METHODS,AGGS,OVERLAP_KINDS,(2,4),(1,2,3)):
            yield 'task_length',seed,dict(method=method,aggregator=agg,
                stream_overlap_kind=kind,scenario='disjoint' if kind=='none' else 'overlap',
                stream_overlap_fraction=0 if kind=='none' else .5,
                classes_per_local_task=classes,local_tasks_per_round=tasks)
        # Exact-duration offline cohorts. D=2 is valid with the default
        # 3-round no-overlap 100-class curriculum. D=3/4 are valid only
        # with expressly configured recurrent rounds.
        for method,agg,kind,rate,duration in product(
                METHODS,AGGS,OVERLAP_KINDS,(.8,.5),(1,2,3,4)):
            if kind=='none' and duration>3:continue
            recur=0 if duration<=2 else 3
            yield 'offline_duration',seed,dict(method=method,aggregator=agg,
                stream_overlap_kind=kind,scenario='disjoint' if kind=='none' else 'overlap',
                stream_overlap_fraction=0 if kind=='none' else .5,
                participation=rate,offline_duration_rounds=duration,
                stream_recurrence_rounds=recur)
        # Distinct label overlap strengths and nonduplicate/raw-duplicate
        # training examples. Cross-client same and later, same-client same/later.
        for method,agg,kind,density in product(METHODS,AGGS,
                OVERLAP_KINDS[1:],(.25,.5,1.)):
            yield 'overlap_strength',seed,dict(method=method,aggregator=agg,
                scenario='overlap',stream_overlap_kind=kind,
                stream_overlap_fraction=density)
        for method,agg,kind,raw in product(METHODS,AGGS,
                ('inter_same_round','inter_cross_round'),(0.,.25,.5)):
            yield 'raw_sample_overlap',seed,dict(method=method,aggregator=agg,
                scenario='overlap',stream_overlap_kind=kind,
                stream_overlap_fraction=.5,sample_overlap_fraction=raw)
        for method,agg,relation in product(METHODS,AGGS,('natural','related','unrelated')):
            yield 'semantic_order',seed,dict(method=method,aggregator=agg,relation=relation)
        for method,agg,power in product(METHODS,AGGS,(0.,.5,1.,2.)):
            yield 'class_imbalance',seed,dict(method=method,aggregator=agg,class_imbalance_power=power)
        for method,agg,skew in product(METHODS,AGGS,(0.,1.,3.)):
            yield 'client_quantity',seed,dict(method=method,aggregator=agg,client_sample_skew=skew)
        # Participation and alternate selection on the shared default stream.
        for method,agg,selector,rate in product(METHODS,AGGS,
                ('random','fixed','round_robin','coverage','least_recent','old','new',
                 'related','recurrence','overlap','balanced','diversity','fairness'),
                (.5,.8,1.)):
            yield 'selection',seed,dict(method=method,aggregator=agg,
                                      selector=selector,participation=rate)
        for method,agg,lr,epochs in product(METHODS,AGGS,(.003,.01,.03,.1),(1,2,4)):
            yield 'local_optimization',seed,dict(method=method,aggregator=agg,lr=lr,epochs=epochs)
        for method,agg,memory in product(
                ('gem','agem','replay','er_ace','der','derpp','lwf'),AGGS,(0,10,20,50,100)):
            yield 'memory',seed,dict(method=method,aggregator=agg,memory_per_class=memory)
        for method,agg,reg in product(('ewc','si','mas','lwf'),AGGS,(.1,1.,10.,100.)):
            yield 'regularization',seed,dict(method=method,aggregator=agg,regularization_strength=reg)
        for method,agg,mu in product(METHODS,AGGS,(0.,.001,.01,.1)):
            yield 'fedprox',seed,dict(method=method,aggregator=agg,fedprox_mu=mu)
        for method,agg,server_lr in product(METHODS,('fedadam','fedavgm'),(.001,.01,.1,.5,1.)):
            yield 'server_optimization',seed,dict(method=method,aggregator=agg,server_lr=server_lr)
        for method,agg,ratio in product(('replay','er_ace','der','derpp'),AGGS,(.25,.5,1.,2.)):
            yield 'replay_ratio',seed,dict(method=method,aggregator=agg,replay_ratio=ratio)
        for method,agg,alpha in product(('der','derpp','lwf'),AGGS,(.1,.3,.5,1.)):
            yield 'distillation',seed,dict(method=method,aggregator=agg,der_alpha=alpha)
        for method,agg,beta in product(('er_ace','derpp'),AGGS,(.1,.3,.5,1.)):
            yield 'replay_beta',seed,dict(method=method,aggregator=agg,der_beta=beta)
        for agg,group in product(AGGS,('class','stage','all')):
            yield 'gem_grouping',seed,dict(method='gem',aggregator=agg,gem_grouping=group)
        for method,agg,n in product(METHODS,AGGS,(5,10,20)):
            yield 'client_count',seed,dict(method=method,aggregator=agg,num_clients=n)
        for method,agg,terminal_flush in product(METHODS,AGGS,(False,True)):
            yield 'terminal_flush',seed,dict(method=method,aggregator=agg,
                stream_terminal_flush=terminal_flush)
        # Explicit recurrence extends disconnected-client histories without
        # artificially claiming additional new classes.
        for method,agg,kind,repeats in product(METHODS,AGGS,
                    ('intra_cross_round','inter_cross_round','combined'),(0,2,4)):
            yield 'recurrent_rounds',seed,dict(method=method,aggregator=agg,
                scenario='overlap',stream_overlap_kind=kind,
                stream_overlap_fraction=.5,stream_recurrence_rounds=repeats)


def expand(dataset='cifar100',seeds=SEEDS,families=None,model='vit',include_invalid=False):
    if dataset!='cifar100':raise ValueError('CIFAR-10 and 20-class experiments have been retired')
    included=set(families) if families else None
    unique=set();bases={seed:base(seed,model=model) for seed in seeds}
    for family,seed,changed in raw_design(seeds):
        if included is not None and family not in included:continue
        # Subset selectors may rely on simulator-privileged metadata; retain
        # explicit oracle warning in the methodology documentation.
        if changed.get('aggregator')=='fedadam' and 'server_lr' not in changed:
            changed={**changed,'server_lr':.001}
        cfg=replace(bases[seed],**changed)
        cfg.validate()
        if not include_invalid:
            reason=check_job(cfg)
            if reason:continue
        signature=fingerprint(cfg)
        if signature in unique:continue
        unique.add(signature)
        cfg=replace(cfg,job_id=f'ST_{family}_{signature[:20]}',block_id='')
        yield family,cfg


def manifest(seeds=SEEDS,families=None,model='vit'):
    counts=Counter();pairs=set();digest=sha256()
    for family,cfg in expand(seeds=seeds,families=families,model=model):
        counts[family]+=1;pairs.add((cfg.method,cfg.aggregator))
        digest.update((cfg.job_id+':'+fingerprint(cfg)+'\n').encode())
    return dict(dataset='cifar100',batch_size=1024,backbone=resolve_backbone(model),job_count=sum(counts.values()),
                families=dict(sorted(counts.items())),method_aggregator_pairs=len(pairs),
                sha256=digest.hexdigest(),seeds=list(seeds),
                warning='These are planned runs, not trained CIFAR-100 results.')


def main(argv=None):
    p=argparse.ArgumentParser(description='CIFAR-100 continuous local-task communication experiments')
    p.add_argument('--dataset',choices=['cifar100'],default='cifar100')
    p.add_argument('--backbone',choices=['vit','cnn'],default='vit')
    p.add_argument('--suite',choices=['full','balanced'],default='full')
    p.add_argument('--families',nargs='*',default=None)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--output-dir',default='results')
    p.add_argument('--data-root',default='data')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--export-plan',default=None)
    p.add_argument('--download',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args(argv)
    seeds=[args.seed] if args.seed is not None else args.seeds
    if len(seeds)!=len(set(seeds)):p.error('Seed values must be distinct')
    families=['balanced'] if args.suite=='balanced' else args.families
    plan=manifest(seeds=seeds,families=families,model=args.backbone)
    root=Path(args.output_dir)/('cifar100_native32_v11_server_fidelity_'+args.backbone);root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'stream_manifest.json',plan)
    print(json.dumps(plan,indent=2),flush=True)
    if args.export_plan:
        destination=Path(args.export_plan);destination.parent.mkdir(parents=True,exist_ok=True)
        opener=gzip.open if str(destination).endswith('.gz') else open
        with opener(destination,'wt') as fp:
            for _,cfg in expand(seeds=seeds,families=families,model=args.backbone):
                fp.write(json.dumps(cfg.to_dict(),sort_keys=True)+'\n')
    if args.plan_only:return
    import torch
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('GPU production requires CUDA; --allow-cpu is test-only')
    attempted=0;failed=[];completed=0
    for index,(family,cfg) in enumerate(expand(seeds=seeds,families=families,model=args.backbone)):
        if index<args.start:continue
        if args.limit and attempted>=args.limit:break
        attempted+=1
        try:
            print(f'[{index+1}/{plan["job_count"]}] {family} {cfg.job_id}',flush=True)
            result=run_experiment(cfg,str(root/'runs'),args.data_root,args.download)
            if result['status']=='completed':completed+=1
        except Exception as exc:
            failed.append(dict(job_id=cfg.job_id,error=repr(exc)))
            print('FAIL:',cfg.job_id,repr(exc),flush=True)
            if not args.continue_on_error:break
        finally:
            atomic_json(root/'stream_suite_status.json',dict(planned=plan['job_count'],
                attempted=attempted,completed=completed,failed=failed,
                manifest_sha256=plan['sha256']))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
