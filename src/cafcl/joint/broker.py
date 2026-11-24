"""Conservative Linux multi-GPU controller: one subprocess per leased GPU.

No GPU memory estimate is invented. The operator supplies a minimum free
memory threshold after measuring representative workloads. Leases coordinate
cooperating controllers on this host; unrelated processes are never killed.
"""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from time import time_ns

from .runner import write_json
from .result_audit import audit_job, validate_manifest
from .curriculum import digest


def utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lease(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Lease already held: '+str(path)) from exc
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def command_text(args):
    return subprocess.check_output(args,text=True).strip()


def gpu_snapshot(gpu):
    raw=command_text(['nvidia-smi','--id='+str(gpu),
        '--query-gpu=uuid,name,memory.free,memory.total','--format=csv,noheader,nounits'])
    rows=raw.splitlines()
    if len(rows)!=1:raise RuntimeError('Expected one physical GPU')
    uuid,name,free,total=[x.strip() for x in rows[0].split(',')]
    apps=command_text(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid',
                       '--format=csv,noheader,nounits'])
    pids=[int(x.strip()) for x in apps.splitlines() if x.strip()]
    return {'gpu':gpu,'uuid':uuid,'name':name,'free_mib':int(free),'total_mib':int(total),'compute_pids':pids}


def gate(snapshot,min_free,allow_shared):
    if snapshot['free_mib']<min_free:return 'insufficient_gpu_memory'
    if snapshot['compute_pids'] and not allow_shared:return 'unrelated_gpu_processes'
    return None


def select_jobs(catalog,family=None,limit=0):
    jobs=[j for j in catalog['jobs'] if not family or family in j.get('families',[j.get('family')])]
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate job IDs')
    for j in jobs:
        if Path(j['job_id']).name!=j['job_id'] or j['job_id'] in ('.','..'):
            raise ValueError('Unsafe job ID')
    # Shorter communication trajectories first; deterministic proxy, not ETA.
    jobs.sort(key=lambda j:(j.get('communication_intervals',0),j.get('task_occurrences',0),j['job_id']))
    return jobs[:limit] if limit else jobs


def run_broker(args, progress_callback=None):
    if len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0 or args.min_free_mib<=0:
        raise ValueError('Distinct nonnegative GPU IDs and positive memory threshold required')
    if args.retries<0 or args.limit<0 or args.concurrency<1 or args.min_disk_gib<0:
        raise ValueError('Invalid resource or retry limits')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=Path(args.manifest).resolve()
    with lease(root/'.joint_broker.lock'):
        catalog=json.loads(manifest.read_text())
        head=command_text(['git','rev-parse','HEAD'])
        if command_text(['git','status','--porcelain','--untracked-files=no']):
            raise RuntimeError('Broker requires a clean tracked source checkout')
        # Dataset downloads happen only in the serial preflight, not workers.
        validation=validate_manifest(manifest,args.data_root,not args.no_download)
        if validation['invalid']:raise RuntimeError('Manifest validation failed: '+repr(validation['invalid'][:3]))
        jobs=select_jobs(catalog,args.family,args.limit)
        excluded=getattr(args,'exclude_family',None)
        if excluded:jobs=[j for j in jobs if excluded not in j.get('families',[j.get('family')])]
        if not jobs:raise ValueError('No matching jobs')
        campaign_id=str(time_ns())
        receipt_root=root/'broker_receipts'/campaign_id;receipt_root.mkdir(parents=True)
        write_json(receipt_root/'campaign.json',{'git_commit':head,'manifest_hash':digest(catalog),
            'gpus':args.gpus,'minimum_free_mib':args.min_free_mib,'min_disk_gib':args.min_disk_gib,
            'job_ids':[j['job_id'] for j in jobs],'started_at':utc(),
            'local_evaluation':args.local_evaluation,'mechanistic_probes':args.mechanistic_probes,'checkpoint_policy':args.checkpoint_policy,'client_concurrency':args.concurrency})
        queue=deque(jobs);lock=Lock();results=[];blocked_gpus={}
        def record(row):
            with lock:
                results.append(row)
                write_json(receipt_root/'progress.json',{'results':results,'blocked_gpus':blocked_gpus})
                if progress_callback:progress_callback(row)
        def worker(gpu):
            try:
                snapshot=gpu_snapshot(gpu)
                # UUID, not index, remains stable across CUDA visibility mappings.
                lease_dir=Path(tempfile.gettempdir())/'cafcl_gpu_leases'
                with lease(lease_dir/(snapshot['uuid']+'.lock')):
                    while True:
                        snap=gpu_snapshot(gpu)
                        reason=gate(snap,args.min_free_mib,args.allow_shared_gpu)
                        if reason:
                            with lock:blocked_gpus[str(gpu)]=reason
                            return
                        if shutil.disk_usage(root).free<args.min_disk_gib*1024**3:
                            with lock:blocked_gpus[str(gpu)]='insufficient_output_disk'
                            return
                        with lock:
                            if not queue:return
                            job=queue.popleft()
                        jid=job['job_id']
                        prior=audit_job(job,root)
                        if prior['status']=='artifact_verified':
                            config=json.loads((root/jid/'config.json').read_text())
                            requested={'joint_local_evaluation':args.local_evaluation,
                                       'joint_mechanistic_probes':args.mechanistic_probes,
                                       'joint_checkpoint_policy':args.checkpoint_policy,
                                       'client_training_concurrency':args.concurrency}
                            if any(config.get(k)!=v for k,v in requested.items()):
                                record({'job_id':jid,'gpu':gpu,'status':'failed','reason':'existing_run_execution_options_differ'})
                            else:record({'job_id':jid,'gpu':gpu,'status':'already_verified'})
                            continue
                        if prior['status']=='audit_failed':
                            record({'job_id':jid,'gpu':gpu,'status':'failed','reason':prior.get('reason')})
                            continue
                        status='failed'
                        for attempt in range(args.retries+1):
                            if command_text(['git','rev-parse','HEAD'])!=head or command_text(['git','status','--porcelain','--untracked-files=no']):
                                raise RuntimeError('Campaign source changed during execution')
                            if digest(json.loads(manifest.read_text()))!=digest(catalog):
                                raise RuntimeError('Campaign manifest changed during execution')
                            log=receipt_root/f'{jid}.attempt{attempt+1:03d}.log'
                            cmd=[sys.executable,'-m','cafcl.joint.runner','run','--manifest',str(manifest),
                                 '--job-id',jid,'--output',str(root),'--data-root',str(Path(args.data_root).resolve()),
                                 '--no-download','--concurrency',str(args.concurrency),'--checkpoint-policy',args.checkpoint_policy]
                            if args.local_evaluation:cmd.append('--local-evaluation')
                            if args.mechanistic_probes:cmd.append('--mechanistic-probes')
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=snapshot['uuid'],
                                     CAFCL_REQUIRE_CUDA='1',CAFCL_EXPECT_GIT_HEAD=head,PYTHONUNBUFFERED='1')
                            started=utc()
                            if progress_callback:
                                with lock:progress_callback({'job_id':jid,'gpu':gpu,'status':'running','attempt':attempt+1,'started_at':started,'log':str(log)})
                            with log.open('w') as handle:
                                process=subprocess.run(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
                            verified=audit_job(job,root)
                            status='verified' if process.returncode==0 and verified['status']=='artifact_verified' else 'failed'
                            write_json(log.with_suffix('.json'),{'job_id':jid,'attempt':attempt+1,'gpu':gpu,
                                'started_at':started,'finished_at':utc(),'exit_code':process.returncode,
                                'gpu_before':snap,'artifact_audit':verified,'status':status})
                            if status=='verified' or verified['status']=='audit_failed':break
                            snap=gpu_snapshot(gpu)
                            if gate(snap,args.min_free_mib,args.allow_shared_gpu):break
                        record({'job_id':jid,'gpu':gpu,'status':status,'attempts':attempt+1,
                                'audit':verified,'log':str(log)})
            except Exception as exc:
                with lock:blocked_gpus[str(gpu)]=repr(exc)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            list(pool.map(worker,args.gpus))
        # Include a popped but unrecorded job if a worker itself failed.
        done={r['job_id'] for r in results}
        blocked=[j['job_id'] for j in jobs if j['job_id'] not in done]
        report={'verified':sum(r['status'] in ('verified','already_verified') for r in results),
                'failed':sum(r['status']=='failed' for r in results),'blocked':blocked,
                'blocked_gpus':blocked_gpus,'results':results,'finished_at':utc(),
                'receipt_directory':str(receipt_root)}
        write_json(receipt_root/'final.json',report)
        if progress_callback:
            for jid in blocked:progress_callback({'job_id':jid,'status':'blocked','reason':'broker_resource_or_worker_blocker'})
        return report
