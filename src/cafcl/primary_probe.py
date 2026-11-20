"""Two concurrent CIFAR-100 Replay/FedAvg arms: full vs random 1/5 offline.

All five clients learn one adjacent two-class task each interval. The pairs
form a chain across clients and intervals, covering 100 classes in 20 learning
syncs. The offline arm randomly omits one upload per sync while all five
clients continue training; a final communication-only flush uploads pending
updates.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
from math import ceil
import numpy as np
from .full_matrix import base
from .models import resolve_backbone
from .streaming import make_task_schedule,select_offline_cohort
from .simulator import run_experiment,atomic_json
from .selection import participation_budget


def make_jobs():
    seed=0;lr=.03;epochs=10;replay_fraction=1/9
    pct_tag='11p1111'
    cfg=replace(base(seed=seed,model='cnn'),num_clients=5,batch_size=64,classwise_head_merge=False,method='replay',replay_reservoir_fraction=replay_fraction,share_continual_state=True,
        aggregator='fedavg',scenario='disjoint',stream_overlap_kind='none',
        stream_overlap_fraction=0.,stream_client_chain=True,
        classes_per_local_task=2,local_tasks_per_round=1,
        stream_recurrence_rounds=0,stream_terminal_flush=True,participation=1.,
        stream_offline_policy='random',evaluate_final_only=True,lr=lr,epochs=epochs,memory_per_class=ceil(450*replay_fraction),
        optimizer_state_policy='persist',client_training_concurrency=5,
        training_augmentation='cifar_crop_flip',
        job_id='PRIMARY_FULL_REPLAY_RESERVOIR'+pct_tag+'_V23_CHAIN20_'+resolve_backbone('cnn'))
    partial=replace(cfg,participation=.8,stream_offline_policy='random',
                    job_id='PRIMARY_RANDOM_OFFLINE_REPLAY_RESERVOIR'+pct_tag+'_V23_CHAIN20_'+resolve_backbone('cnn'))
    cfg=replace(cfg,job_id='PRIMARY_FULL_REPLAY_RESERVOIR'+pct_tag+'_V23_CHAIN20_'+resolve_backbone('cnn'))
    for job in (cfg,partial):job.validate()
    return cfg,partial


def audited_plan(jobs):
    baseline,partial=jobs
    stream=make_task_schedule(100,baseline)
    rounds=1+max(r for r,_,_,_,_ in stream)
    assert rounds==20 and len(stream)==5*20
    assert all(task==0 for _,_,task,_,_ in stream)
    assert all(len(labels)==2 for _,_,_,labels,_ in stream)
    by_client_round={(r,cid):labels for r,cid,_,labels,_ in stream}
    assert set(by_client_round)=={(r,cid) for r in range(20) for cid in range(5)}
    for r in range(20):
        for cid in range(4):
            # The final client slot repeats (98,99) so all five clients can
            # train at sync 20 without introducing an out-of-range class.
            if (r,cid)==(19,3):
                assert by_client_round[r,cid+1]==by_client_round[r,cid]
            else:
                assert by_client_round[r,cid][1]==by_client_round[r,cid+1][0]
        if r<19:
            assert by_client_round[r,4][1]==by_client_round[r+1,0][0]
    flattened=[label for _,_,_,labels,_ in stream for label in labels]
    occurrences={cls:flattened.count(cls) for cls in range(100)}
    assert set(flattened)==set(range(100))
    assert occurrences[0]==1 and occurrences[98]==3
    assert all(occurrences[c]==2 for c in range(1,100) if c!=98)
    assert all(by_client_round[0,cid]==(cid,cid+1) for cid in range(5))
    assert all(by_client_round[1,cid]==(cid+5,cid+6) for cid in range(5))
    assert by_client_round[19,3]==(98,99) and by_client_round[19,4]==(98,99)
    from .selection import select_clients
    rng=np.random.default_rng(partial.seed+888_777)
    states={cid:None for cid in range(partial.num_clients)}
    offline=[]
    for r in range(rounds):
        standard=select_clients(states,partial,r,rng)
        selected=select_offline_cohort(partial,r,rounds+1,standard)
        absent=set(range(5))-set(selected)
        assert len(absent)==1
        offline.append(next(iter(absent)))
    by_round=[]
    for r in range(rounds):
        by_round.append(dict(round=r+1,offline_client=offline[r],
            offline_classes=list(by_client_round[r,offline[r]]),
            online_clients=[c for c in range(5) if c!=offline[r]]))
        assert len(by_round[-1]['offline_classes'])==2
    source_fedclass=baseline.model=='fedclass_resnet8'
    per_class_source=(baseline.memory_per_class if source_fedclass and baseline.method=='replay'
                      else ceil(450*baseline.replay_reservoir_fraction))
    source_selection=('authors_original_first_k' if source_fedclass and baseline.method=='replay'
                      else 'none' if source_fedclass else 'independent_classwise_algorithm_r')
    plan=dict(dataset='cifar100',seed=baseline.seed,backbone=baseline.model,batch_size=baseline.batch_size,sgd_momentum=baseline.momentum,sgd_weight_decay=baseline.weight_decay,
        client_execution_mode=('parallel_task_barriers' if not baseline.client_training_concurrency
                               else 'bounded_concurrent_task_barriers'),
        client_models_per_run=baseline.num_clients,
        canonical_whole_model_aggregation=not baseline.classwise_head_merge,
        process_topology='two_independent_runs_on_one_gpu',
        replay_reservoir_fraction=baseline.replay_reservoir_fraction,
        replay_sampling=source_selection,
        replay_update='each incoming batch plus every unique currently available archive example before one optimizer step',
        replay_microbatch_size=baseline.batch_size,
        archive_broadcast='complete aggregate replay archive to all clients each sync; model only to selected clients',
        source_memory_count_per_class=per_class_source,
        training_augmentation=baseline.training_augmentation,
        optimizer_state_policy=baseline.optimizer_state_policy,
        training_images_per_class=450,private_reservoir_per_class=per_class_source,
        expected_server_exemplars_after_flush=sum(occurrences.values())*per_class_source,
        expected_server_exemplars_per_class={str(c):occurrences[c]*per_class_source
                                              for c in range(100)},
        clients=5,classes_per_local_task=2,initial_local_tasks=1,
        subsequent_local_tasks=1,initial_new_classes_per_client=2,
        new_unique_classes_first_sync=6,new_unique_classes_syncs_2_to_19=5,
        new_unique_classes_final_sync=4,
        local_tasks_per_client_per_sync=1,training_intervals=rounds,
        terminal_communication_only_flush=True,all_classes_covered=True,
        offline_selection='seeded random one-client outage independently each learning sync',
        all_clients_train_each_sync=True,all_clients_receive_replay_archive_each_sync=True,
        selected_clients_receive_model_each_sync=True,
        server_models_per_run=1,
        models_per_run=6,server_model_updates='only after the five-client learning barrier',
        experiment_processes_concurrent=2,
        server_models_across_both_arms=2,client_models_across_both_arms=10,
        concurrently_training_client_models_across_both_arms=10,
        class_occurrence_counts={str(c):occurrences[c] for c in range(100)},
        terminal_pair_repeated_for_full_20_sync_coverage=[98,99],
        total_jobs=2,job_ids=[c.job_id for c in jobs],offline_schedule=by_round)
    plan['client_training_concurrency']=(
        baseline.client_training_concurrency or baseline.num_clients)
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
        rows.append(dict(class_id=c,full_accuracy=a,random_offline_accuracy=b,delta=b-a))
    overall=dict(full_accuracy=float(full['accuracy']),random_offline_accuracy=float(partial['accuracy']),
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
    markdown=['# CIFAR-100 Replay/FedAvg: full participation versus random one-client outage','',
        'Two matched runs, five clients, one adjacent two-class task/client at every learning sync,',
        '20 learning syncs cover all 100 labels, followed by one communication-only flush.',
        'All five client models train each sync; the hidden arm randomly omits one upload per sync.',
        'Two independent experiment processes run concurrently on one GPU.',
        'The terminal schedule repeats pair 98,99 for the fifth client so all five train in sync 20.',
        f"Training augmentation: {jobs[0].training_augmentation}; optimizer-state policy: {jobs[0].optimizer_state_policy}.",
        'Per-sync held-out scores are REPORTING ONLY: never used by client selection/training.',
        'Per-sync CSV, PNG, JSON and MD tables: runs/<job_id>/per_sync_accuracy/sync_###.*', '',
        f"- Full accuracy: {overall['full_accuracy']:.6f}",
        f"- Random-offline accuracy: {overall['random_offline_accuracy']:.6f}",
        f"- Difference (offline minus full): {overall['delta']:+.6f}",
        '- Both models integrated all 100 labels: yes','',
        '| Class | Full | Random offline | Delta |','|---:|---:|---:|---:|']
    markdown.extend(f"| {r['class_id']} | {r['full_accuracy']:.4f} | {r['random_offline_accuracy']:.4f} | {r['delta']:+.4f} |" for r in rows)
    (folder/'comparison.md').write_text('\n'.join(markdown)+'\n')
    return overall


def main(argv=None):
    p=argparse.ArgumentParser(description='Two simultaneous 5-client chain-stream simulations sharing ONE GPU')
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-dir',default='results')
    p.add_argument('--gpu',type=int,default=0,
                   help='One physical GPU shared by BOTH concurrent runs (default: 0)')
    p.add_argument('--gpus',default=None,
                   help='Legacy spelling, must specify the SAME GPU twice, e.g. 0,0')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--worker',type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    jobs=make_jobs()
    design=audited_plan(jobs)
    root=Path(args.output_dir)/'primary_cifar100_fedavg_v23_resnet18_cropflip_exhaustive_replay_chain20_randomoffline_bs64_seed0'
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'design.json',design)
    if args.plan_only:
        print(json.dumps(design,indent=2));return
    if args.worker is not None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required; check CUDA_VISIBLE_DEVICES and PyTorch CUDA installation')
        result=run_experiment(jobs[args.worker],str(root/'runs'),args.data_root,download=False)
        print(json.dumps(result,indent=2),flush=True);return
    # Download/cache ONCE, before both processes access the immutable dataset.
    from .data import get_data_bank
    get_data_bank('cifar100',args.data_root,download=not args.no_download,seed=0)
    available=[str(args.gpu),str(args.gpu)]
    if args.gpus is not None:
        alias=[x.strip() for x in args.gpus.split(',')]
        if len(alias)!=2 or not all(x.isdigit() for x in alias) or alias[0]!=alias[1]:
            p.error('Only one GPU is supported: use --gpu 0 or --gpus 0,0; do not use 0,1')
        available=alias
    cmd=[sys.executable,'-m','cafcl.primary_probe',
         '--data-root',args.data_root,
         '--output-dir',args.output_dir,'--no-download']
    processes=[]
    for idx,gpu in enumerate(available):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        env.setdefault('OMP_NUM_THREADS','1')
        env.setdefault('MKL_NUM_THREADS','1')
        log=root/(f'job_{idx}.log')
        handle=log.open('a')
        proc=subprocess.Popen(cmd+['--worker',str(idx)],env=env,stdout=handle,stderr=subprocess.STDOUT)
        processes.append((proc,handle))
        print(f'Launched {jobs[idx].job_id}: PID={proc.pid}, GPU={gpu}, log={log}',flush=True)
    codes=[]
    for proc,handle in processes:
        codes.append(proc.wait())
        handle.close()
    if any(c!=0 for c in codes):
        raise RuntimeError(f'One or both pilot jobs failed: {codes}. Inspect per-job logs; no false success reported.')
    report=build_report(root,jobs)
    if report is None:
        raise RuntimeError('A worker reported success without both final summary files')
    print('FINAL PAIRED RESULT',json.dumps(report,indent=2),flush=True)
    print('Report:',root/'comparison'/'comparison.md',flush=True)

if __name__=='__main__':main()
