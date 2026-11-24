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


def plan_campaign(output,data_root,seed,download,depths=(1,2,3,5),suite='foundational'):
    from ..data import get_data_bank
    bank=get_data_bank('cifar100',data_root,download=download,seed=seed)
    root=Path(output).resolve()
    jobs=[]
    if suite in ('screening','interactions','short','extensions','controls','all'):
        from .family_catalog import screening_designs,Design
        from .interactions import interaction_designs,short_designs
        generators=[]
        if suite in ('screening','all'):generators.append(screening_designs())
        if suite in ('interactions','all'):generators.append(interaction_designs())
        if suite in ('short','all'):generators.append(short_designs())
        if suite in ('extensions','all'):
            from .extensions import extension_designs
            generators.append(extension_designs())
            from .distribution_designs import distribution_designs
            generators.append(distribution_designs(bank.coarse_by_fine))
        if suite=='all':
            generators.append((Design(f'H-{s}-O{o}-L{l}','H',foundational_stream(s,o),local_depth=l)
                for s in ('S1','S2','S3') for o in (0,1) for l in depths))
        if suite in ('controls','all'):
            from .controls import control_designs
            generators.append(control_designs())
        blocked=[]
        for design in chain.from_iterable(generators):
            try:
                plan=compile_plan(design.events,bank.train.targets,num_clients=design.num_clients,
                    num_classes=100,local_depth=design.local_depth,seed=seed,
                    head_policy=design.head_policy,minimum_samples=1,
                    require_all_classes=design.require_all_classes,
                    interval_boundaries=design.boundaries,upload_cohorts=design.cohorts,
                    offline_clients=design.offline_clients,outage_duration=design.outage_duration,
                    allocation_order=design.allocation_order,
                    raw_sample_overlap=design.raw_sample_overlap,
                    allow_cross_semantic_reuse=design.allow_cross_semantic_reuse,
                    allocation_weights=design.allocation_weights,control_kind=design.control_kind,
                    terminal_flush=design.terminal_flush).to_dict()
            except ValueError as exc:
                blocked.append({'design':design.name,'reason':str(exc),'status':'infeasible'})
                continue
            # Head-only interventions have no meaning for a shared Class-IL
            # classifier. Do not spend compute on duplicate Class-IL controls.
            settings=('task_il',) if design.family=='short' and design.head_policy!='semantic_shared' else ('task_il','class_il')
            for setting in settings:
                job_id='J-'+design.name+'-'+setting
                filename=job_id+'.json';write_json(root/'plans'/filename,plan)
                jobs.append({'job_id':job_id,'family':design.family,'metadata':design.metadata,
                    'learning_setting':setting,'epochs':design.epochs,'method':design.method,'domain_mode':design.domain_mode,'seed':seed,'plan_file':'plans/'+filename,
                    'allow_partial_coverage':not design.require_all_classes,
                    'plan_hash':plan['plan_hash'],'atomic_hash':plan['atomic_hash'],'status':'planned',
                    'low_support_warning':plan['allocation_audit']['minimum_samples_per_class_occurrence']<8})
        unique={};aliases=[]
        for job in jobs:
            key=(job['plan_hash'],job['learning_setting'],job['epochs'],job.get('method','replay'),job.get('domain_mode','none'))
            if key in unique:
                aliases.append({'job_id':job['job_id'],'canonical_job_id':unique[key]['job_id']})
            else:unique[key]=job
        jobs=list(unique.values())
        write_json(root/'manifest.json',{'dataset':'cifar100','jobs':jobs,'aliases':aliases,'infeasible':blocked,
                   'implementation_blocked':['joint intra-task synchronization; compute-matched pooled controls']})
        return jobs
    for item in foundational_registry(depths,seed=seed):
        events=foundational_stream(item['spatial'],item['local_overlap'])
        plan=compile_plan(events,bank.train.targets,num_clients=5,num_classes=100,
                          local_depth=item['local_depth'],head_policy=item['head_policy'],
                          seed=seed,minimum_samples=8).to_dict()
        filename=item['job_id']+'.json'
        write_json(root/'plans'/filename,plan)
        jobs.append({**item,'plan_file':'plans/'+filename,'plan_hash':plan['plan_hash'],
                     'atomic_hash':plan['atomic_hash'],'status':'planned',
                     'task_occurrences':len(plan['tasks']),
                     'communication_intervals':len(plan['intervals'])})
    write_json(root/'manifest.json',{'dataset':'cifar100','jobs':jobs})
    return jobs


def run_job(manifest,job_id,output,data_root,download,concurrency=1,local_evaluation=False):
    from ..config import RunConfig
    from ..simulator import run_experiment
    manifest=Path(manifest).resolve()
    jobs=json.loads(manifest.read_text())['jobs']
    found=[x for x in jobs if x['job_id']==job_id]
    if len(found)!=1:
        raise ValueError('Expected one matching job')
    job=found[0];path=(manifest.parent/job['plan_file']).resolve()
    plan=json.loads(path.read_text())
    counts=Counter()
    lookup={x['occurrence_id']:x for x in plan['tasks']}
    for r in plan['intervals']:
        for oid in r['occurrences']:
            counts[(r['interval_id'],lookup[oid]['client_id'])]+=1
    cfg=RunConfig(dataset='cifar100',model='resnet18_cifar32',stream_protocol=True,
        num_clients=plan['num_clients'],epochs=job['epochs'],batch_size=64,
        participation=1.,selector='random',scenario='overlap',
        stages=1,rounds_per_stage=1,learning_setting=job['learning_setting'],
        local_tasks_per_round=max(counts.values()),method=job.get('method','replay'),
        aggregator='fedavg',memory_per_class=50 if job.get('method','replay')=='replay' else 0,classwise_head_merge=False,
        replay_ratio=1. if job.get('method','replay')=='replay' else 0.,training_augmentation='cifar_crop_flip',
        sample_overlap_fraction=plan['allocation_audit']['raw_sample_overlap'],
        client_training_concurrency=concurrency,seed=job['seed'],
        joint_plan_path=str(path),joint_plan_hash=job['plan_hash'],
        joint_local_evaluation=local_evaluation,
        joint_allow_partial_coverage=job.get('allow_partial_coverage',False),
        joint_control_kind=plan['allocation_audit'].get('control_kind','federated'),
        joint_domain_mode=job.get('domain_mode','none'),
        stream_terminal_flush=plan['intervals'][-1]['terminal_flush'],
        checkpoint_every=1,job_id=job_id)
    job_dir=Path(output)/job_id
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):head=None;dirty=None
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
    a.add_argument('--suite',choices=['foundational','screening','interactions','short','extensions','controls','all'],default='foundational')
    a.add_argument('--depths',type=int,nargs='+',default=[1,2,3,5])
    a=sub.add_parser('run');a.add_argument('--manifest',required=True)
    a.add_argument('--job-id',required=True);a.add_argument('--output',required=True)
    a.add_argument('--concurrency',type=int,default=1)
    a.add_argument('--local-evaluation',action='store_true')
    a.add_argument('--gpu',type=int)
    a=sub.add_parser('run-suite');a.add_argument('--manifest',required=True)
    a.add_argument('--output',required=True);a.add_argument('--family')
    a.add_argument('--concurrency',type=int,default=1);a.add_argument('--gpu',type=int)
    a.add_argument('--local-evaluation',action='store_true')
    a=sub.add_parser('validate');a.add_argument('--manifest',required=True)
    a=sub.add_parser('audit');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('report');a.add_argument('--run-dir',required=True);a.add_argument('--output')
    a.add_argument('--no-figures',action='store_true')
    for a in sub.choices.values():
        a.add_argument('--data-root',default='data')
        a.add_argument('--no-download',action='store_true')
    a=p.parse_args(argv)
    if a.command=='plan':
        jobs=plan_campaign(a.output,a.data_root,a.seed,not a.no_download,a.depths,a.suite)
        print(f'Planned {len(jobs)} conditions; no training launched')
    elif a.command=='validate':
        from .result_audit import validate_manifest
        report=validate_manifest(a.manifest,a.data_root,not a.no_download)
        print(json.dumps(report,indent=2))
        if report['invalid']:raise SystemExit(1)
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
                          not a.no_download,a.concurrency,a.local_evaluation))
        else:
            # Sequential controller: resource-broker integration is intentionally
            # deferred rather than oversubscribing available GPUs.
            jobs=json.loads(Path(a.manifest).read_text())['jobs']
            failures=[]
            for job in jobs:
                if a.family and job.get('family')!=a.family:continue
                try:print(run_job(a.manifest,job['job_id'],a.output,a.data_root,
                              not a.no_download,a.concurrency,a.local_evaluation),flush=True)
                except Exception as exc:
                    failures.append({'job_id':job['job_id'],'error':repr(exc)})
                    write_json(Path(a.output)/'campaign_failures.json',failures)
            if failures:raise SystemExit(1)


if __name__=='__main__':
    main()
