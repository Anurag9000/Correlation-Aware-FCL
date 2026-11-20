"""Precisely two CIFAR-100 Replay/FedAvg pilot jobs: full vs rotating 1/5 offline.

Each job has five clients. Initially each client learns two consecutive
two-class experiences, then one two-class experience per further round.
Nine training intervals cover all 100 distinct classes. The rotating
offline client cycles across all five; terminal flush uploads pending state.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
from math import ceil
from .full_matrix import base
from .models import resolve_backbone
from .streaming import make_task_schedule,select_offline_cohort
from .simulator import run_experiment,atomic_json
from .selection import participation_budget


def make_jobs(*,backbone='vit',seed=0,lr=.03,epochs=10,replay_fraction=1/9):
    pct_tag=f'{100*replay_fraction:g}'.replace('.','p')
    cfg=replace(base(seed=seed,model=backbone),num_clients=5,batch_size=64,classwise_head_merge=False,method='replay',replay_reservoir_fraction=replay_fraction,share_continual_state=True,
        aggregator='fedavg',scenario='disjoint',stream_overlap_kind='none',
        stream_overlap_fraction=0.,classes_per_local_task=2,local_tasks_per_round=2,
        stream_recurrence_rounds=0,stream_terminal_flush=True,participation=1.,
        stream_offline_policy='random',evaluate_final_only=True,lr=lr,epochs=epochs,memory_per_class=ceil(450*replay_fraction),
        job_id='PRIMARY_FULL_REPLAY_RESERVOIR'+pct_tag+'_V16_BATCH64_'+resolve_backbone(backbone))
    partial=replace(cfg,participation=.8,stream_offline_policy='rotate_once',
                    job_id='PRIMARY_ROTATING_OFFLINE_REPLAY_RESERVOIR'+pct_tag+'_V16_BATCH64_'+resolve_backbone(backbone))
    for job in (cfg,partial):job.validate()
    return cfg,partial


def audited_plan(jobs):
    baseline,partial=jobs
    stream=make_task_schedule(100,baseline)
    rounds=1+max(r for r,_,_,_,_ in stream)
    assert rounds==9 and len(stream)==5*(2+8)
    flattened=[label for _,_,_,labels,_ in stream for label in labels]
    assert len(flattened)==100 and len(set(flattened))==100 and set(flattened)==set(range(100))
    assert all(len(labels)==2 for _,_,_,labels,_ in stream)
    standard=list(range(partial.num_clients-1))
    offline=[]
    for r in range(rounds):
        selected=select_offline_cohort(partial,r,rounds+1,standard)
        absent=set(range(5))-set(selected)
        assert len(absent)==1
        offline.append(next(iter(absent)))
    assert set(offline)==set(range(5)) and len(offline)==9
    by_round=[]
    for r in range(rounds):
        by_round.append(dict(round=r+1,offline_client=offline[r],
            offline_classes=sorted({c for rr,cid,_,labels,_ in stream if rr==r and cid==offline[r] for c in labels}),
            online_clients=[c for c in range(5) if c!=offline[r]]))
        assert len(by_round[-1]['offline_classes'])==(4 if r==0 else 2)
    plan=dict(dataset='cifar100',seed=baseline.seed,backbone=baseline.model,batch_size=baseline.batch_size,sgd_momentum=baseline.momentum,sgd_weight_decay=baseline.weight_decay,
        client_execution_mode=('parallel_task_barriers' if not baseline.client_training_concurrency
                               else 'bounded_concurrent_task_barriers'),
        client_models_per_run=baseline.num_clients,
        canonical_whole_model_aggregation=not baseline.classwise_head_merge,
        process_topology='two_independent_runs_on_one_gpu',
        replay_reservoir_fraction=baseline.replay_reservoir_fraction,
        replay_sampling='independent_classwise_algorithm_r',
        training_images_per_class=450,private_reservoir_per_class=ceil(450*baseline.replay_reservoir_fraction),
        expected_server_exemplars_after_flush=100*ceil(450*baseline.replay_reservoir_fraction),
        clients=5,classes_per_local_task=2,initial_local_tasks=2,
        subsequent_local_tasks=1,initial_new_classes_per_client=4,
        subsequent_new_classes_per_client=2,training_intervals=rounds,
        terminal_communication_only_flush=True,all_classes_once=True,
        offline_rotation_covers_all=True,
        total_jobs=2,job_ids=[c.job_id for c in jobs],offline_schedule=by_round)
    if baseline.client_training_concurrency:
        plan['client_training_concurrency']=baseline.client_training_concurrency
    return plan


def build_report(root,jobs):
    summaries=[]
    for c in jobs:
        path=root/'runs'/c.job_id/'summary.json'
        if not path.exists():return None
        summaries.append(json.loads(path.read_text()))
    full,partial=[x['final_metrics'] for x in summaries]
    if set(map(int,full['per_class']))!=set(range(100)) or set(map(int,partial['per_class']))!=set(range(100)):
        raise RuntimeError('Pilot final evaluation must cover all 100 class IDs')
    rows=[]
    for c in range(100):
        a=float(full['per_class'][str(c)])
        b=float(partial['per_class'][str(c)])
        rows.append(dict(class_id=c,full_accuracy=a,rotating_offline_accuracy=b,delta=b-a))
    overall=dict(full_accuracy=float(full['accuracy']),rotating_offline_accuracy=float(partial['accuracy']),
                 delta=float(partial['accuracy'])-float(full['accuracy']),
                 final_integrated_full=len(full['globally_integrated_classes']),
                 final_integrated_offline=len(partial['globally_integrated_classes']))
    expected=set(range(100))
    if set(map(int,full['globally_integrated_classes']))!=expected or \
       set(map(int,partial['globally_integrated_classes']))!=expected:
        raise RuntimeError('Final integrated CIFAR-100 label IDs must be exactly 0..99')
    folder=root/'comparison';folder.mkdir(exist_ok=True,parents=True)
    atomic_json(folder/'summary.json',overall)
    atomic_json(folder/'per_class.json',rows)
    with (folder/'per_class.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    markdown=['# CIFAR-100 preliminary probe: full participation versus rotating one-client outage','',
        'Two matched Replay/FedAvg runs, five clients, first round two binary tasks/client,',
        'then one binary task/client per round for eight more rounds (nine total),',
        'one terminal communication-only flush, seed controlled, original 100-class test split.',
        'Per-sync held-out scores are REPORTING ONLY: never used by client selection/training.',
        'Per-sync CSV, PNG, JSON and MD tables: runs/<job_id>/per_sync_accuracy/sync_###.*', '',
        f"- Full accuracy: {overall['full_accuracy']:.6f}",
        f"- Rotating-offline accuracy: {overall['rotating_offline_accuracy']:.6f}",
        f"- Difference (offline minus full): {overall['delta']:+.6f}",
        '- Both models integrated all 100 labels: yes','',
        '| Class | Full | Rotating offline | Delta |','|---:|---:|---:|---:|']
    markdown.extend(f"| {r['class_id']} | {r['full_accuracy']:.4f} | {r['rotating_offline_accuracy']:.4f} | {r['delta']:+.4f} |" for r in rows)
    (folder/'comparison.md').write_text('\n'.join(markdown)+'\n')
    return overall


def main(argv=None):
    p=argparse.ArgumentParser(description='Two simultaneous 5-client task-barrier simulations sharing ONE GPU')
    p.add_argument('--backbone',choices=('vit','cnn'),default='vit')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--replay-fraction',type=float,default=1/9,help='Per-class fraction of available TRAIN examples retained by Algorithm R')
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-dir',default='results')
    p.add_argument('--gpu',type=int,default=0,
                   help='One physical GPU shared by BOTH concurrent runs (default: 0)')
    p.add_argument('--gpus',default=None,
                   help='Legacy spelling, must specify the SAME GPU twice, e.g. 0,0')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--allow-cpu',action='store_true',help='Synthetic/debug only; does not make a GPU production run')
    p.add_argument('--worker',type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    jobs=make_jobs(backbone=args.backbone,seed=args.seed,lr=args.lr,epochs=args.epochs,
                   replay_fraction=args.replay_fraction)
    design=audited_plan(jobs)
    root=Path(args.output_dir)/('primary_cifar100_batch64_v16_'+args.backbone+
        ('' if abs(args.replay_fraction-.1)<1e-12 else '_pct'+f'{100*args.replay_fraction:g}'.replace('.','p'))+
        '_bs64_seed'+str(args.seed))
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'design.json',design)
    if args.plan_only:
        print(json.dumps(design,indent=2));return
    if args.worker is not None:
        import torch
        if not args.allow_cpu and not torch.cuda.is_available():
            raise RuntimeError('GPU required; check CUDA_VISIBLE_DEVICES and PyTorch CUDA installation')
        result=run_experiment(jobs[args.worker],str(root/'runs'),args.data_root,download=False)
        print(json.dumps(result,indent=2),flush=True);return
    # Download/cache ONCE, before both processes access the immutable dataset.
    from .data import get_data_bank
    get_data_bank('cifar100',args.data_root,download=not args.no_download,seed=args.seed)
    available=[str(args.gpu),str(args.gpu)]
    if args.gpus is not None:
        alias=[x.strip() for x in args.gpus.split(',')]
        if len(alias)!=2 or not all(x.isdigit() for x in alias) or alias[0]!=alias[1]:
            p.error('Only one GPU is supported: use --gpu 0 or --gpus 0,0; do not use 0,1')
        available=alias
    cmd=[sys.executable,'-m','cafcl.primary_probe','--backbone',args.backbone,
         '--seed',str(args.seed),'--lr',str(args.lr),'--epochs',str(args.epochs),
         '--replay-fraction',str(args.replay_fraction),
         '--data-root',args.data_root,
         '--output-dir',args.output_dir,'--no-download']
    if args.allow_cpu:cmd.append('--allow-cpu')
    processes=[]
    try:
        for idx,gpu in enumerate(available):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            env.setdefault('OMP_NUM_THREADS','1')
            env.setdefault('MKL_NUM_THREADS','1')
            log=root/(f'job_{idx}.log')
            handle=log.open('a')
            proc=subprocess.Popen(cmd+['--worker',str(idx)],env=env,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((proc,handle))
            print(f'Launched {jobs[idx].job_id}: PID={proc.pid}, GPU={gpu}, log={log}',flush=True)
        codes=[proc.wait() for proc,_ in processes]
    finally:
        for proc,handle in processes:
            if proc.poll() is None:proc.terminate()
            handle.close()
    if any(c!=0 for c in codes):
        raise RuntimeError(f'One or both pilot jobs failed: {codes}. Inspect per-job logs; no false success reported.')
    report=build_report(root,jobs)
    if report is None:
        raise RuntimeError('A worker reported success without both final summary files')
    print('FINAL PAIRED RESULT',json.dumps(report,indent=2),flush=True)
    print('Report:',root/'comparison'/'comparison.md',flush=True)

if __name__=='__main__':main()
