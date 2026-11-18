"""Finite-grid, fully crossed CIFAR FCL research suites.

Preserves legacy A--N plans. The declared grids are comprehensive across
implemented method/server pairs but are not a claim that continuous spaces,
missing algorithms, official-test-safe tuning, or training are completed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time

from .planner import all_supported_suite, base_config
from .simulator import atomic_json, run_experiment

METHODS = ('sgd','gem','agem','replay','ewc','si','er_ace','der','derpp','mas','lwf')
AGGS = ('fedavg','uniform','fedavgm','fedadam','class_balanced',
        'inverse_participation','conflict_aware','conflict_projection','retention_aware')
MAIN = ('disjoint','overlap','related','recurrence')
POLICIES = (('random',1.),('random',.5),('random',.25),('coverage',.5),('coverage',.25))
SELECTORS = ('random','fixed','round_robin','coverage','old','new','least_recent',
             'related','recurrence','overlap','balanced','diversity','fairness',
             'stochastic_related','historical_coverage','feature_max','feature_min',
             'gradient_align','gradient_conflict','forgetting','gem_informed','history_adaptive')
OTHER = ('iid','non_iid','asynchronous','mixed','stationary','domain',
         'gradual_drift','shifted_recurrence','task_incremental')
MEMORY = ('gem','agem','replay','er_ace','der','derpp','lwf')
REGULARIZED = ('ewc','si','mas','lwf')
SEEDS = (0,1,2)


def raw_grids(dataset, seeds):
    """Yield (family, kwargs, seed) in immutable, scientifically named order."""
    for seed in seeds:
        def grid(family, dimensions, fixed=None):
            keys = tuple(dimensions)
            for values in product(*(dimensions[k] for k in keys)):
                yield family, {**dict(zip(keys,values)),**(fixed or {})}, seed
        # Full 11 x 9 x 4 x 5 cross for each random realization.
        for m,a,s,(selector,rate) in product(METHODS,AGGS,MAIN,POLICIES):
            yield 'balanced',dict(method=m,aggregator=a,scenario=s,
                                  selector=selector,participation=rate),seed
        yield from grid('scenarios', dict(method=METHODS,aggregator=AGGS,scenario=OTHER))
        yield from grid('selectors', dict(method=METHODS,aggregator=AGGS,
                   scenario=('related','overlap','recurrence'),selector=SELECTORS))
        yield from grid('participation',dict(method=METHODS,aggregator=AGGS,
                   scenario=('overlap','related'),
                   selector=('random','fixed','round_robin','coverage','least_recent'),
                   participation=(.1,.25,.5,.75,1.)))
        yield from grid('overlap',dict(method=METHODS,aggregator=AGGS,
                   overlap=(0.,.25,.5,.75,1.),selector=('random','coverage')),
                   dict(scenario='overlap'))
        yield from grid('sample_overlap',dict(method=METHODS,aggregator=AGGS,
                   sample_overlap_fraction=(0.,.25,.5,.75,1.)),dict(scenario='overlap'))
        yield from grid('dirichlet',dict(method=METHODS,aggregator=AGGS,
                   dirichlet_alpha=(.05,.1,.5,1.,10.)),dict(scenario='non_iid'))
        yield from grid('local_optimization',dict(method=METHODS,aggregator=AGGS,
                   lr=(.003,.01,.03,.1),epochs=(1,2,4),scenario=('disjoint','related')))
        yield from grid('memory',dict(method=MEMORY,aggregator=AGGS,
                   memory_per_class=(0,10,20,50,100),scenario=('disjoint','recurrence')))
        yield from grid('regularization',dict(method=REGULARIZED,aggregator=AGGS,
                   regularization_strength=(.1,1.,10.,100.),scenario=('disjoint','related')))
        yield from grid('fedprox',dict(method=METHODS,aggregator=AGGS,
                   fedprox_mu=(0.,.001,.01,.1),scenario=('disjoint','non_iid')))
        yield from grid('server_optimization',dict(method=METHODS,
                   aggregator=('fedadam','fedavgm'),server_lr=(.001,.01,.1,.5,1.),
                   scenario=('disjoint','related')))
        yield from grid('recurrence_timing',dict(method=METHODS,aggregator=AGGS,
                   recurrence_round_delay=(1,5,15,30),
                   recurrence_client_policy=('same','different')),dict(scenario='recurrence'))
        yield from grid('imbalance',dict(method=METHODS,aggregator=AGGS,
                   class_imbalance_power=(0.,.5,1.,2.)),dict(scenario='overlap'))
        yield from grid('quantity_skew',dict(method=METHODS,aggregator=AGGS,
                   client_sample_skew=(0.,1.,3.)),dict(scenario='overlap'))
        yield from grid('backbone',dict(method=METHODS,aggregator=AGGS,
                   model=('linear','mlp','smallcnn'),scenario=('disjoint','related')))
        yield from grid('client_population',dict(method=METHODS,aggregator=AGGS,
                   num_clients=(5,10,20),scenario=('overlap','related')))
        yield from grid('local_momentum',dict(method=METHODS,aggregator=AGGS,
                   momentum=(0.,.9),optimizer_state_policy=('reset','persist')),
                   dict(scenario='disjoint'))
        yield from grid('memory_policy',dict(method=MEMORY,aggregator=AGGS,
                   memory_policy=('random_per_class','recent_per_class')),
                   dict(scenario='overlap'))
        yield from grid('replay_ratio',dict(method=('replay','er_ace','der','derpp'),
                   aggregator=AGGS,replay_ratio=(.25,.5,1.,2.),
                   scenario=('disjoint','related')))
        yield from grid('distillation_alpha',dict(method=('der','derpp','lwf'),
                   aggregator=AGGS,der_alpha=(.1,.3,.5,1.),
                   scenario=('disjoint','related')))
        yield from grid('replay_beta',dict(method=('er_ace','derpp'),aggregator=AGGS,
                   der_beta=(.1,.3,.5,1.),scenario=('disjoint','related')))
        yield from grid('si_damping',dict(aggregator=AGGS,si_xi=(.01,.1,1.),
                   scenario=('disjoint','related')),dict(method='si'))
        yield from grid('server_betas',dict(method=METHODS,server_beta1=(.5,.9,.99),
                   server_beta2=(.9,.99,.999),scenario=('disjoint','related')),
                   dict(aggregator='fedadam',server_lr=.001))
        yield from grid('gem_constraints',dict(aggregator=AGGS,
                   gem_grouping=('class','stage','all'),scenario=('disjoint','related')),
                   dict(method='gem'))
        for family,d,seed0 in grid('pending_data',dict(method=METHODS,aggregator=AGGS,
                   pending_policy=('latest','retain_all','bounded'),
                   scenario=('asynchronous','recurrence'))):
            d['max_pending_samples']=400 if d['pending_policy']=='bounded' else 0
            yield family,d,seed0
        for family,d,seed0 in grid('participation_schedule',dict(method=METHODS,
                   aggregator=AGGS,participation_schedule=('constant','increasing',
                   'decreasing','alternating','stage','feedback'),
                   scenario=('overlap','related'))):
            if d['participation_schedule']=='stage':
                d['stage_participation']=(.25,.75) if dataset=='cifar10' else (.25,)+(.75,)*9
            yield family,d,seed0
        yield from grid('target_client_interventions',dict(method=METHODS,aggregator=AGGS,
                   force_target=('include','exclude'),scenario=('related','recurrence')),
                   dict(target_client=2,intervention_round=20))
        for family,d,seed0 in grid('spurious_features',dict(method=METHODS,
                   aggregator=AGGS,spurious_strength=(0.,1.)),dict(scenario='related')):
            d['spurious_ood_eval']=bool(d['spurious_strength'])
            yield family,d,seed0
        yield from grid('label_noise',dict(scenario=('disjoint','non_iid'),
                   label_noise_rate=(0.,.1,.2)),
                   dict(method='sgd',aggregator='fedavg',selector='random'))
        yield from grid('heterogeneous_clients',dict(aggregator=AGGS,
                   scenario=('disjoint','related'),client_heterogeneity=(False,True)),
                   dict(method='sgd'))
        if dataset=='cifar100':
            for m,a,s,(selector,rate) in product(METHODS,AGGS,
                            ('related','recurrence','overlap'),
                            (('random',.5),('coverage',.5))):
                yield 'cifar100_20', dict(dataset='cifar100_20',stages=2,
                       method=m,aggregator=a,scenario=s,selector=selector,
                       participation=rate),seed


def fingerprint(cfg):
    payload=asdict(cfg);payload.pop('job_id',None);payload.pop('block_id',None)
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def expand(dataset,seeds=SEEDS,include_legacy=True,families=None):
    if dataset not in ('cifar10','cifar100'):raise ValueError(dataset)
    if include_legacy:
        for cfg in all_supported_suite(dataset):yield 'legacy',cfg
    seen=set();selected=set(families) if families else None
    base={s:base_config(dataset,seed=s) for s in seeds}
    for family,changes,seed in raw_grids(dataset,seeds):
        if selected is not None and family not in selected:continue
        if changes.get('aggregator')=='fedadam' and 'server_lr' not in changes:
            changes={**changes,'server_lr':.001}
        cfg=replace(base[seed],**changes)
        digest=fingerprint(cfg)
        if digest in seen:continue
        seen.add(digest)
        cfg=replace(cfg,job_id=f'FX_{family}_{digest[:20]}',block_id='')
        cfg.validate()
        if cfg.batch_size != 512:raise RuntimeError('CIFAR batch size must be 512')
        yield family,cfg


def plan(dataset,seeds=SEEDS,include_legacy=True,families=None):
    counts=Counter();pairs=set();h=sha256()
    for family,cfg in expand(dataset,seeds,include_legacy,families):
        counts[family]+=1;pairs.add((cfg.method,cfg.aggregator))
        h.update((cfg.job_id+':'+fingerprint(cfg)+'\n').encode())
    return dict(dataset=dataset,seeds=list(seeds),batch_size=512,
        total_jobs=sum(counts.values()),family_counts=dict(sorted(counts.items())),
        method_server_pairs=len(pairs),possible_pairs=99,
        manifest_sha256=h.hexdigest(),
        warning='Finite grid of planned jobs, not executed CIFAR results or complete paper reproductions.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('cifar10','cifar100'),required=True)
    p.add_argument('--suite',choices=('full','balanced','factorial','legacy'),default='full')
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--seed',type=int)
    p.add_argument('--families',default='')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--export-plan',default='')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--start',type=int,default=1)
    p.add_argument('--output-dir',default='results');p.add_argument('--data-root',default='data')
    p.add_argument('--download',action='store_true');p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--continue-on-error',action='store_true');p.add_argument('--stop-on-error',action='store_true')
    args=p.parse_args()
    seeds=(args.seed,) if args.seed is not None else tuple(int(x) for x in args.seeds.split(','))
    if not seeds or len(seeds)!=len(set(seeds)):p.error('seeds must be distinct')
    families=tuple(x for x in args.families.split(',') if x) or None
    if args.suite=='balanced':families=('balanced',)
    if args.suite=='legacy':families=('__no_factorial__',)
    legacy=args.suite in ('full','balanced','legacy')
    root=Path(args.output_dir)/args.dataset;root.mkdir(parents=True,exist_ok=True)
    info=plan(args.dataset,seeds,legacy,families)
    planfile=root/f'{args.suite}_plan_summary.json'
    if planfile.exists() and json.loads(planfile.read_text())['manifest_sha256'] != info['manifest_sha256']:
        raise RuntimeError('Plan mismatch: use another RESULTS_ROOT for different axes/seeds')
    atomic_json(planfile,info)
    print(json.dumps(info,indent=2),flush=True)
    if args.export_plan:
        import gzip
        path=Path(args.export_plan);path.parent.mkdir(parents=True,exist_ok=True)
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'wt',encoding='utf8') as fp:
            for ix,(family,cfg) in enumerate(expand(args.dataset,seeds,legacy,families),1):
                fp.write(json.dumps(dict(index=ix,family=family,config=cfg.to_dict()))+'\n')
    if args.plan_only:return
    if not args.allow_cpu:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing silent CPU fallback')
    if args.start<1 or args.limit<0:raise ValueError('Invalid start/limit')
    import fcntl
    attempts=completed=skipped=0;failed=[]
    with (root/'.suite.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,(family,cfg) in enumerate(expand(args.dataset,seeds,legacy,families),1):
            if index<args.start:continue
            if args.limit and attempts>=args.limit:break
            attempts+=1
            print(f'[{index}/{info["total_jobs"]}] {cfg.job_id} {family}',flush=True)
            try:
                source=None
                if cfg.fork_source_job:
                    source=root/'runs'/cfg.fork_source_job/f'checkpoint_round_{cfg.fork_checkpoint_round:04d}.pt'
                    if not source.is_file():raise RuntimeError(f'Missing fork dependency: {source}')
                res=run_experiment(cfg,str(root/'runs'),args.data_root,args.download,
                                   fork_from_checkpoint=source)
                rd=root/'runs'/cfg.job_id
                report=json.loads((rd/'summary.json').read_text())
                if not args.allow_cpu and report.get('device')!='cuda':
                    raise RuntimeError('Expected CUDA model training')
                if json.loads((rd/'config.json').read_text()) != cfg.to_dict():
                    raise RuntimeError('Artifact/config mismatch')
                for name in ('status.json','metrics.json','model_final.pt'):
                    if not (rd/name).exists():raise RuntimeError('Missing output '+name)
                skipped+=res['status']=='already_completed'
                completed+=res['status']!='already_completed'
            except Exception as exc:
                failed.append({'index':index,'job_id':cfg.job_id,'error':repr(exc)})
            atomic_json(root/f'{args.suite}_status.json',dict(total_jobs=info['total_jobs'],
                last_attempted=index,completed=completed,skipped=skipped,failed=failed,
                manifest_sha256=info['manifest_sha256'],updated=time.time()))
            if failed and not args.continue_on_error:break
    print(f'completed={completed}, skipped={skipped}, failed={len(failed)}')
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
