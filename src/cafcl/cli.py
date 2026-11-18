"""Local execution and sequential all-supported experiment orchestration."""
from __future__ import annotations
import argparse,json,traceback,time,os,subprocess
from dataclasses import replace
from pathlib import Path
from .config import RunConfig
from .planner import core_suite,all_supported_suite,write_plan
from .simulator import run_experiment,atomic_json
from .catalogue import CATALOGUE,IMPLEMENTED,catalogue_report


def select_jobs(jobs, requested_ids):
    """Select named jobs while preserving their validated plan order."""
    if not requested_ids:
        return list(jobs)
    requested=list(requested_ids)
    if len(requested)!=len(set(requested)):
        raise ValueError('Requested job IDs must be unique')
    by_id={job.job_id:job for job in jobs}
    missing=sorted(set(requested)-set(by_id))
    if missing:
        raise ValueError(f'Unknown job IDs: {missing}')
    selected=set(requested)
    return [job for job in jobs if job.job_id in selected]


def wait_for_host_memory(minimum_gib, poll_seconds=30):
    """Pause between jobs until the shared host has the requested headroom."""
    if minimum_gib <= 0:
        return
    minimum_kib=int(minimum_gib*1024*1024)
    while True:
        available_kib=None
        with open('/proc/meminfo',encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    available_kib=int(line.split()[1])
                    break
        if available_kib is None:
            raise RuntimeError('Cannot read MemAvailable for the suite resource gate')
        if available_kib>=minimum_kib:
            return
        print(f'Waiting for host memory headroom: {available_kib/1024/1024:.1f} GiB '
              f'available; minimum is {minimum_gib:.1f} GiB',flush=True)
        time.sleep(max(5,poll_seconds))


def wait_for_exclusive_gpu(current_pid, poll_seconds=30):
    """Pause suite progression while another non-allowlisted CUDA job is active."""
    allowlisted={int(value) for value in os.environ.get(
        'CAFCL_GPU_CPU_ONLY_PIDS','').split() if value.isdigit()}
    while True:
        probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                              '--format=csv,noheader,nounits'],
                             capture_output=True,text=True)
        if probe.returncode:
            raise RuntimeError(f'Cannot verify exclusive GPU ownership: {probe.stderr.strip()}')
        pids=sorted({int(value.strip()) for value in probe.stdout.splitlines()
                     if value.strip().isdigit()}-{current_pid}-allowlisted)
        if not pids:return
        print(f'Waiting for other GPU compute processes to exit: {pids}',flush=True)
        time.sleep(max(5,poll_seconds))


def main():
    p=argparse.ArgumentParser(prog='cafcl')
    sp=p.add_subparsers(dest='action',required=True)
    suite=sp.add_parser('suite',help='Plan or run sequential CIFAR experiments')
    suite.add_argument('--dataset',choices=['cifar10','cifar100','synthetic'],required=True)
    suite.add_argument('--suite',choices=['core','all-supported'],default='all-supported')
    suite.add_argument('--output-dir',default='results')
    suite.add_argument('--data-root',default='data')
    suite.add_argument('--seed',type=int,default=0)
    suite.add_argument('--plan-only',action='store_true')
    suite.add_argument('--download',action='store_true',help='Download official CIFAR datasets if unavailable')
    suite.add_argument('--stop-on-error',action='store_true')
    suite.add_argument('--limit',type=int,default=0,help='First N jobs for smoke testing')
    suite.add_argument('--job-ids',nargs='+',default=[],
                       help='Run these planned job IDs in canonical plan order')
    suite.add_argument('--include-incomplete',action='store_true',help='Fail if any planned block is unimplemented')
    suite.add_argument('--wait-for-exclusive-gpu',action='store_true',
                       help='Wait between jobs while another CUDA compute process is active')
    suite.add_argument('--gpu-poll-seconds',type=int,default=30)
    suite.add_argument('--minimum-memory-gib',type=float,default=0.0,
                       help='Wait between jobs until MemAvailable reaches this value')
    single=sp.add_parser('run',help='Run one experiment configuration JSON')
    single.add_argument('config');single.add_argument('--output-dir',default='results')
    single.add_argument('--data-root',default='data');single.add_argument('--download',action='store_true')
    fork=sp.add_parser('fork',help='Full multi-round counterfactual fork from a saved checkpoint')
    fork.add_argument('--config',required=True,help='JSON baseline configuration')
    fork.add_argument('--checkpoint',required=True,help='Validated same-scenario checkpoint')
    fork.add_argument('--client',type=int,required=True)
    fork.add_argument('--arm',choices=['include','exclude'],required=True)
    fork.add_argument('--round',type=int,required=True)
    fork.add_argument('--job-id',required=True)
    fork.add_argument('--output-dir',default='results/counterfactuals')
    fork.add_argument('--data-root',default='data')
    fork.add_argument('--download',action='store_true')
    cat=sp.add_parser('catalogue',help='Print implemented and missing block IDs')
    cat.add_argument('--output',default='')
    args=p.parse_args()
    if args.action=='catalogue':
        report=catalogue_report()
        if args.output:Path(args.output).write_text(json.dumps(report,indent=2))
        else:print(json.dumps(report,indent=2))
        return
    if args.action=='fork':
        baseline=RunConfig(**json.loads(Path(args.config).read_text()))
        config=replace(baseline,job_id=args.job_id,target_client=args.client,
                       force_target=args.arm,intervention_round=args.round)
        print(json.dumps(run_experiment(config,args.output_dir,args.data_root,
                                        args.download,fork_from_checkpoint=args.checkpoint),indent=2))
        return
    if args.action=='run':
        config=RunConfig(**json.loads(Path(args.config).read_text()))
        print(json.dumps(run_experiment(config,args.output_dir,args.data_root,args.download),indent=2))
        return
    if args.action=='suite':
        if args.include_incomplete and len(IMPLEMENTED)<len(CATALOGUE):
            raise SystemExit(f'Cannot claim all 152 implemented: only {len(IMPLEMENTED)} '
                             f'blocks have any implementation, some partial.')
        jobs=core_suite(args.dataset,args.seed) if args.suite=='core' else all_supported_suite(args.dataset,args.seed)
        root=Path(args.output_dir)/args.dataset
        root.mkdir(parents=True,exist_ok=True)
        plan=write_plan(root/'job_plan.json',jobs,args.dataset)
        registered_job_count=len(jobs)
        jobs=select_jobs(jobs,args.job_ids)
        jobs_to_run=jobs[:args.limit] if args.limit else jobs
        print(f'Prepared {len(jobs)} sequential {args.dataset} jobs, '
              f'{len(IMPLEMENTED)}/{len(CATALOGUE)} block IDs with implementation coverage.',flush=True)
        if args.job_ids:
            print(f'Selected {len(jobs_to_run)} named jobs from {registered_job_count} planned jobs.',flush=True)
        if args.plan_only:return
        failures=[];completed=0;skipped=0
        for i,cfg in enumerate(jobs_to_run,1):
            wait_for_host_memory(args.minimum_memory_gib,args.gpu_poll_seconds)
            if args.wait_for_exclusive_gpu:
                wait_for_exclusive_gpu(os.getpid(),args.gpu_poll_seconds)
            print(f'\n=== [{i}/{len(jobs_to_run)}] {cfg.job_id} ({args.dataset}, batch 512) ===',flush=True)
            try:
                source=None
                if cfg.fork_source_job:
                    source=(root/'runs'/cfg.fork_source_job/
                            f'checkpoint_round_{cfg.fork_checkpoint_round:04d}.pt')
                    if not source.is_file():
                        raise RuntimeError(f'Dependent fork checkpoint missing: {source}')
                result=run_experiment(cfg,str(root/'runs'),args.data_root,args.download,
                                      fork_from_checkpoint=source)
                if result['status']=='already_completed':skipped+=1
                else:completed+=1
            except Exception as exc:
                failures.append({'job_id':cfg.job_id,'error':repr(exc)})
                print(f'FAILED {cfg.job_id}: {exc}',flush=True)
                if args.stop_on_error:break
            atomic_json(root/'suite_status.json',{
                'dataset':args.dataset,'planned':len(jobs_to_run),
                'registered_plan_jobs':registered_job_count,
                'selected_job_ids':[job.job_id for job in jobs_to_run],
                'attempted':i,
                'completed_this_invocation':completed,'already_completed':skipped,
                'failures':failures,'batch_size':512,
                'catalogue_unimplemented':[k for k in CATALOGUE if k not in IMPLEMENTED]})
        print(f'Finished invocation: completed={completed}, already_done={skipped}, failed={len(failures)}')
        if failures:raise SystemExit(1)

if __name__=='__main__':main()
