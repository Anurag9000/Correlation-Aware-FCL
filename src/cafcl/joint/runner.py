"""Plan and execute explicit joint experiments through cafcl.simulator.

Examples:
  python -m cafcl.joint.runner plan --output runs/joint_plan
  python -m cafcl.joint.runner run --manifest runs/joint_plan/manifest.json \
      --job-id J-H-S1-O0-L2-task_il --output runs/joint_results
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import chain
import os
import subprocess
import json
from pathlib import Path
from .curriculum import compile_plan, foundational_registry, foundational_stream


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def iter_designs(suite,depths,coarse_by_fine):
    from .family_catalog import screening_designs,Design
    from .interactions import interaction_designs,short_designs
    if suite=='essential':
        from .essential import essential_designs
        yield from essential_designs()
        return
    if suite in ('screening','all'):yield from screening_designs()
    if suite in ('interactions','all'):yield from interaction_designs()
    if suite in ('short','all'):yield from short_designs()
    if suite in ('extensions','all'):
        from .extensions import extension_designs
        from .distribution_designs import distribution_designs
        yield from extension_designs()
        yield from distribution_designs(coarse_by_fine)
    if suite in ('foundational','all'):
        for s in ('S1','S2','S3'):
            for o in (0,1):
                for depth in depths:
                    yield Design(f'H-{s}-O{o}-L{depth}','H',foundational_stream(s,o),local_depth=depth)
    if suite in ('controls','all'):
        from .controls import control_designs
        yield from control_designs()
    if suite in ('intratask','all'):
        from .intratask import intratask_designs
        yield from intratask_designs()
    if suite=='smoke':
        from .smoke import smoke_designs
        yield from smoke_designs()


def compile_design(design,targets,seed):
    return compile_plan(design.events,targets,num_clients=design.num_clients,
        num_classes=100,local_depth=design.local_depth,seed=seed,
        head_policy=design.head_policy,minimum_samples=8 if design.family=='H' and design.name.startswith('H-') else 1,
        require_all_classes=design.require_all_classes,interval_boundaries=design.boundaries,
        upload_cohorts=design.cohorts,offline_clients=design.offline_clients,
        outage_duration=design.outage_duration,allocation_order=design.allocation_order,
        raw_sample_overlap=design.raw_sample_overlap,allow_cross_semantic_reuse=design.allow_cross_semantic_reuse,
        allocation_weights=design.allocation_weights,control_kind=design.control_kind,
        terminal_flush=design.terminal_flush,epoch_segments=design.epoch_segments).to_dict()


def settings_for(design):
    return ('task_il',) if design.family=='short' and design.head_policy!='semantic_shared' else ('task_il','class_il')


def plan_campaign(output,data_root,seed,download,depths=(1,2,3,5),suite='foundational'):
    if suite=='essential' and seed!=0:
        raise ValueError('The essential campaign uses seed 0 only')
    from ..data import get_data_bank
    bank=get_data_bank('cifar100',data_root,download=download,seed=seed)
    root=Path(output).resolve()
    jobs=[]
    unique={};aliases=[];blocked=[];nominal=Counter();written=set()
    for design in iter_designs(suite,depths,bank.coarse_by_fine):
        settings=settings_for(design);nominal[design.family]+=len(settings)
        try:plan=compile_design(design,bank.train.targets,seed)
        except ValueError as exc:
            blocked.append({'design':design.name,'family':design.family,'conditions':len(settings),
                            'reason':str(exc),'status':'infeasible'})
            continue
        for setting in settings:
            job_id='J-'+design.name+'-'+setting
            key=(plan['plan_hash'],setting,design.epochs,design.method,design.domain_mode)
            if key in unique:
                canonical=unique[key]
                if design.family not in canonical['families']:canonical['families'].append(design.family)
                aliases.append({'job_id':job_id,'family':design.family,
                                'canonical_job_id':canonical['job_id'],'metadata':design.metadata})
                continue
            filename=plan['plan_hash']+'.json'
            if filename not in written:
                write_json(root/'plans'/filename,plan);written.add(filename)
            job={'job_id':job_id,'family':design.family,'families':[design.family],
                 'metadata':design.metadata,'learning_setting':setting,'epochs':design.epochs,
                 'method':design.method,'domain_mode':design.domain_mode,'seed':seed,
                 'plan_file':'plans/'+filename,'allow_partial_coverage':not design.require_all_classes,
                 'plan_hash':plan['plan_hash'],'atomic_hash':plan['atomic_hash'],'status':'planned',
                 'task_occurrences':len(plan['tasks']),'communication_intervals':len(plan['intervals']),
                 'low_support_warning':plan['allocation_audit']['minimum_samples_per_class_occurrence']<8}
            unique[key]=job;jobs.append(job)
    if suite=='essential' and (len(jobs)!=60 or aliases or blocked):
        raise RuntimeError(f'Essential campaign must compile exactly 60 unique jobs: {len(jobs)} jobs, {len(aliases)} aliases, {blocked!r}')
    write_json(root/'manifest.json',{'dataset':'cifar100','suite':suite,'jobs':jobs,'aliases':aliases,
               'infeasible':blocked,'nominal_by_family':dict(nominal),
               'nominal_conditions':sum(nominal.values()),'unique_executable_conditions':len(jobs),
               'aliased_conditions':len(aliases),'infeasible_conditions':sum(x['conditions'] for x in blocked),
               'implementation_blocked':['minibatch-level joint synchronization; compute-matched pooled controls']})
    return jobs

def job_config(job,plan,path,concurrency=1,local_evaluation=False,mechanistic_probes=False,checkpoint_policy='all'):
    from ..config import RunConfig
    job_id=job['job_id']
    counts=Counter()
    lookup={x['occurrence_id']:x for x in plan['tasks']}
    for r in plan['intervals']:
        for oid in r['occurrences']:
            counts[(r['interval_id'],lookup[oid]['client_id'])]+=1
    cfg=RunConfig(dataset='cifar100',model='resnet18_cifar32',stream_protocol=True,
        num_clients=plan['num_clients'],epochs=job['epochs'],batch_size=job.get('batch_size',64),
        physical_batch_policy=job.get('physical_batch_policy','fixed_pad'),
        participation=1.,selector='random',scenario='overlap',
        stages=1,rounds_per_stage=1,learning_setting=job['learning_setting'],
        local_tasks_per_round=max(counts.values()),method=job.get('method','replay'),
        aggregator='fedavg',memory_per_class=50 if job.get('method','replay')=='replay' else 0,classwise_head_merge=False,
        replay_ratio=job.get('replay_ratio',1. if job.get('method','replay')=='replay' else 0.),training_augmentation='cifar_crop_flip',
        replay_update_policy=job.get('replay_update_policy','sampled_mixed'),
        replay_class_sampling_policy=job.get('replay_class_sampling_policy','distinct_first'),
        share_continual_state=job.get('share_continual_state',True),
        server_memory_per_class=job.get('server_memory_per_class',0),
        stale_update_policy=job.get('stale_update_policy','damped_delta'),
        terminal_stale_guard=job.get('terminal_stale_guard',True),
        aggregation_weight_policy=job.get('aggregation_weight_policy','fresh_examples'),
        optimizer_state_policy=job.get('optimizer_state_policy','persist'),
        sample_overlap_fraction=plan['allocation_audit']['raw_sample_overlap'],
        client_training_concurrency=concurrency,seed=job['seed'],
        joint_plan_path=str(path),joint_plan_hash=job['plan_hash'],
        joint_local_evaluation=local_evaluation,joint_mechanistic_probes=mechanistic_probes,
        joint_checkpoint_policy=checkpoint_policy,
        joint_allow_partial_coverage=job.get('allow_partial_coverage',False),
        joint_control_kind=plan['allocation_audit'].get('control_kind','federated'),
        joint_domain_mode=job.get('domain_mode','none'),
        joint_epoch_segments=plan['allocation_audit'].get('epoch_slicing',{}).get('segments',1),
        stream_terminal_flush=plan['intervals'][-1]['terminal_flush'],
        checkpoint_every=1,job_id=job_id)
    return cfg


def run_job(manifest,job_id,output,data_root,download,concurrency=1,local_evaluation=False,mechanistic_probes=False,checkpoint_policy='all'):
    from ..config import RunConfig
    from ..simulator import run_experiment
    if os.environ.get('CAFCL_REQUIRE_CUDA')=='1':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU worker cannot access CUDA; refusing CPU fallback')
    manifest=Path(manifest).resolve()
    catalog=json.loads(manifest.read_text());jobs=catalog['jobs']
    alias={a['job_id']:a['canonical_job_id'] for a in catalog.get('aliases',[])}
    job_id=alias.get(job_id,job_id)
    found=[x for x in jobs if x['job_id']==job_id]
    if len(found)!=1:
        raise ValueError('Expected one matching job')
    job=found[0];path=(manifest.parent/job['plan_file']).resolve()
    plan=json.loads(path.read_text())
    cfg=job_config(job,plan,path,concurrency,local_evaluation,mechanistic_probes,checkpoint_policy)
    from .checkpoint_io import job_lease
    with job_lease(Path(output)/job_id):
        return execute_locked(cfg,job,output,data_root,download,run_experiment)


def execute_locked(cfg,job,output,data_root,download,run_experiment):
    job_id=job['job_id'];job_dir=Path(output)/job_id
    if (job_dir/'config.json').exists():
        existing=json.loads((job_dir/'config.json').read_text())
        expected=json.loads(json.dumps(cfg.to_dict(),default=str))
        if existing!=expected:raise RuntimeError('Existing joint run configuration differs; use a new output root')
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):head=None;dirty=None
    required_head=os.environ.get('CAFCL_EXPECT_GIT_HEAD')
    if required_head and (head!=required_head or dirty):
        raise RuntimeError('Worker source revision differs from clean campaign checkout')
    # Preserve the first launch receipt; retries have separate receipts.
    from time import time_ns
    write_json(job_dir/'launch_receipts'/f'{time_ns()}.json',
               {'git_commit':head,'tracked_worktree_dirty':dirty,'job_id':job_id,
                'plan_hash':job['plan_hash'],'config_signature':cfg.signature()})
    return run_experiment(cfg,str(output),data_root,download=download)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan');a.add_argument('--output',required=True)
    a.add_argument('--seed',type=int,default=0)
    a.add_argument('--suite',choices=['foundational','screening','interactions','short','extensions','controls','intratask','smoke','essential','all'],default='essential')
    a.add_argument('--depths',type=int,nargs='+',default=[1,2,3,5])
    a=sub.add_parser('run');a.add_argument('--manifest',required=True)
    a.add_argument('--job-id',required=True);a.add_argument('--output',required=True)
    a.add_argument('--concurrency',type=int,default=1)
    a.add_argument('--local-evaluation',action='store_true');a.add_argument('--mechanistic-probes',action='store_true');a.add_argument('--checkpoint-policy',choices=['all','latest'],default='all')
    a.add_argument('--gpu',type=int)
    a=sub.add_parser('run-suite');a.add_argument('--manifest',required=True)
    a.add_argument('--output',required=True);a.add_argument('--family')
    a.add_argument('--concurrency',type=int,default=1);a.add_argument('--gpu',type=int)
    a.add_argument('--local-evaluation',action='store_true');a.add_argument('--mechanistic-probes',action='store_true');a.add_argument('--checkpoint-policy',choices=['all','latest'],default='all')
    a=sub.add_parser('broker');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a.add_argument('--gpus',type=int,nargs='+',required=True);a.add_argument('--min-free-mib',type=int,required=True)
    a.add_argument('--min-disk-gib',type=float,default=20);a.add_argument('--retries',type=int,default=0)
    a.add_argument('--family');a.add_argument('--limit',type=int,default=0)
    a.add_argument('--concurrency',type=int,default=1);a.add_argument('--local-evaluation',action='store_true');a.add_argument('--mechanistic-probes',action='store_true');a.add_argument('--checkpoint-policy',choices=['all','latest'],default='all')
    a.add_argument('--allow-shared-gpu',action='store_true')
    a=sub.add_parser('campaign',help='Run the complete reduced 60-job campaign')
    from .campaign import add_arguments
    add_arguments(a)
    a=sub.add_parser('validate');a.add_argument('--manifest',required=True)
    a=sub.add_parser('audit');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('compare');a.add_argument('--campaign',nargs=2,action='append',metavar=('MANIFEST','RESULTS'),required=True)
    a.add_argument('--output',required=True);a.add_argument('--contrasts')
    a=sub.add_parser('report');a.add_argument('--run-dir',required=True);a.add_argument('--output')
    a.add_argument('--no-figures',action='store_true')
    for a in sub.choices.values():
        a.add_argument('--data-root',default='data')
        a.add_argument('--no-download',action='store_true')
    a=p.parse_args(argv)
    if a.command=='plan':
        jobs=plan_campaign(a.output,a.data_root,a.seed,not a.no_download,a.depths,a.suite)
        print(f'Planned {len(jobs)} conditions; no training launched')
    elif a.command=='campaign':
        from .campaign import run_campaign
        result=run_campaign(a)
        print(json.dumps(result,indent=2))
        if not result['complete']:raise SystemExit(1)
    elif a.command=='broker':
        from .broker import run_broker
        result=run_broker(a)
        print(json.dumps(result,indent=2))
        if result['failed'] or result['blocked']:raise SystemExit(1)
    elif a.command=='validate':
        from .result_audit import validate_manifest
        report=validate_manifest(a.manifest,a.data_root,not a.no_download)
        print(json.dumps(report,indent=2))
        if report['invalid']:raise SystemExit(1)
    elif a.command=='compare':
        from .comparison import compare_campaigns
        print(json.dumps(compare_campaigns(a.campaign,a.output,a.contrasts),indent=2))
    elif a.command=='report':
        from .analysis import analyze_run
        print(json.dumps(analyze_run(a.run_dir,a.output,not a.no_figures),indent=2))
    elif a.command=='audit':
        from .result_audit import audit_campaign
        print(json.dumps(audit_campaign(a.manifest,a.output),indent=2))
    else:
        if a.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        if a.command=='run':
            print(run_job(a.manifest,a.job_id,a.output,a.data_root,
                          not a.no_download,a.concurrency,a.local_evaluation,a.mechanistic_probes,a.checkpoint_policy))
        else:
            # Sequential controller: resource-broker integration is intentionally
            # deferred rather than oversubscribing available GPUs.
            jobs=json.loads(Path(a.manifest).read_text())['jobs']
            failures=[]
            for job in jobs:
                if a.family and a.family not in job.get('families',[job.get('family')]):continue
                try:print(run_job(a.manifest,job['job_id'],a.output,a.data_root,
                              not a.no_download,a.concurrency,a.local_evaluation,a.mechanistic_probes,a.checkpoint_policy),flush=True)
                except Exception as exc:
                    failures.append({'job_id':job['job_id'],'error':repr(exc)})
                    write_json(Path(a.output)/'campaign_failures.json',failures)
            if failures:raise SystemExit(1)


if __name__=='__main__':
    main()
