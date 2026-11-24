#!/usr/bin/env python3
"""Worker and safe sequential queue controller for independent CAFCL A-G runs."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MIN_FREE_MIB=24576
WAIT_SECONDS=300


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')
    os.replace(tmp,path)


def canonical_json_sha256(obj):
    payload=json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def source_manifest(runtime_root):
    """Hash the exact worker, compiler, and isolated Python source used by a run."""
    roots=[Path(__file__).resolve(),Path(__file__).resolve().with_name('cafcl_independent_ablation_compiler.py'),
           runtime_root/'src'/'cafcl']
    files=[]
    for root in roots:
        candidates=[root] if root.is_file() else sorted(root.rglob('*.py'))
        for path in candidates:
            if '__pycache__' in path.parts or path.suffix!='.py':continue
            relative=(path.name if path.is_file() and path.parent==Path(__file__).resolve().parent
                      else str(path.relative_to(runtime_root)) if path.is_relative_to(runtime_root)
                      else path.name)
            files.append((relative,path))
    entries=[];overall=hashlib.sha256()
    for relative,path in sorted(files,key=lambda row:row[0]):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({'path':relative,'sha256':digest})
        overall.update(relative.encode());overall.update(b'\0');overall.update(digest.encode());overall.update(b'\n')
    return {'algorithm':'sha256','tree_sha256':overall.hexdigest(),'files':entries}


def environment_manifest(runtime_root,cfg):
    import numpy as np
    import torch
    cuda=torch.cuda.is_available()
    return {'hostname':platform.node(),'platform':platform.platform(),
        'python_version':sys.version,'python_executable':sys.executable,
        'numpy_version':np.__version__,'torch_version':torch.__version__,
        'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
        'cuda_available':cuda,'cuda_device_name':torch.cuda.get_device_name(0) if cuda else None,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'runtime_root':str(runtime_root),'dataset':'cifar100','seed':cfg.seed,
        'model':cfg.model,'captured_at':time.time()}


def load_spec(spec_dir):
    spec_dir=Path(spec_dir)
    cfg=json.loads((spec_dir/'config.json').read_text())
    tasks=json.loads((spec_dir/'schedule.json').read_text())
    validation=json.loads((spec_dir/'plan_validation.json').read_text())
    return cfg,tasks,validation


def build_config(cfg_data,tasks):
    from dataclasses import replace
    from cafcl.full_matrix import base
    from cafcl.config import RunConfig
    family=cfg_data['family']; setting=cfg_data.get('learning_setting','task_il')
    n=int(cfg_data.get('num_clients',5))
    width=int(cfg_data.get('classes_per_task_target',cfg_data.get('classes_per_task',2)))
    participation=0.8 if family=='C' else 1.0
    d=int(cfg_data.get('offline_duration_rounds',0))
    explicit=[];seen=set()
    for task in tasks:
        labels=tuple(map(int,task['classes']))
        recurrent=any(x in seen for x in labels)
        seen.update(labels)
        explicit.append((int(task['sync']),int(task['client']),0,labels,recurrent))
    cfg=replace(base(seed=0,model='cnn'),
        dataset='cifar100',batch_size=64,num_clients=n,rounds_per_stage=1,epochs=10,
        participation=participation,selector='random',method='replay',aggregator='fedavg',
        scenario='disjoint',memory_per_class=50,replay_reservoir_fraction=1/9,
        share_continual_state=True,lr=0.03,seed=0,full_sync=False,
        replay_ratio=1.0,replay_update_policy='sampled_mixed',
        optimizer_state_policy='persist',momentum=0.9,weight_decay=5e-4,
        job_id=f"CAFCL_{family}_{cfg_data['run_id'] if 'run_id' in cfg_data else 'seed0'}",
        block_id=family,learning_setting=setting,class_pair_protocol='strict',
        classwise_head_merge=False,training_augmentation='cifar_crop_flip',
        stream_protocol=True,classes_per_local_task=width,local_tasks_per_round=1,
        stream_client_chain=False,stream_overlap_kind='none',stream_overlap_fraction=0.0,
        sample_overlap_fraction=0.0,stream_recurrence_rounds=0,
        offline_duration_rounds=d,stream_terminal_flush=True,
        stream_offline_policy='random',stream_sync_every_batch=False,
        stream_partial_batch_policy='include_padding',evaluate_final_only=False,
        client_training_concurrency=min(5,n),resume=False,
        stream_schedule_override=tuple(explicit))
    if not isinstance(cfg,RunConfig): raise TypeError('Unexpected RunConfig constructor result')
    cfg.validate()
    return cfg,explicit


def worker(spec_dir,plan_only=False):
    cfg_data,tasks,expected=load_spec(spec_dir)
    cfg_data['run_id']=expected['run_id']
    cfg,explicit=build_config(cfg_data,tasks)
    from cafcl.streaming import make_task_schedule, compile_stream_scenario
    compiled=make_task_schedule(100,cfg)
    if compiled!=explicit:
        raise RuntimeError(f"Runner schedule adapter disagrees with saved schedule for {expected['run_id']}")
    if plan_only:
        from collections import Counter
        from cafcl.data import get_data_bank
        from cafcl.streaming import select_offline_cohort
        bank=get_data_bank('cifar100',str(ROOT/f'runtime_baseline_{"taskil" if cfg.learning_setting=="task_il" else "classil"}'/'data'),download=False,seed=0)
        scenario=compile_stream_scenario(bank,cfg)
        if len(scenario.arrivals)!=len(tasks): raise RuntimeError('Arrival/task count mismatch')
        if set(scenario.label_holders)!=set(range(100)): raise RuntimeError('Not all CIFAR-100 labels are exposed')
        if scenario.duplicate_sample_ids: raise RuntimeError('Schedule reuses raw training images')
        if cfg.learning_setting=='task_il':
            ids=[a.task_id for a in scenario.arrivals]
            if len(ids)!=len(set(ids)): raise RuntimeError('Task-IL task IDs are not unique')
        task_sample_counts=[len(a.indices) for a in scenario.arrivals]
        class_occurrences=Counter(x for task in tasks for x in task['classes'])
        outage_validated=True
        if cfg_data['family']=='C':
            planned={int(row['sync']):row['uploaded_clients'] for row in cfg_data['participation_schedule']}
            rounds=max(a.round_id for a in scenario.arrivals)+1
            for r in range(rounds):
                selected=select_offline_cohort(cfg,r,rounds+1,list(range(cfg.num_clients-1)))
                if selected!=planned[r]: raise RuntimeError(f'Outage cohort mismatch at sync {r}: {selected} != {planned[r]}')
        report={"run_id":expected['run_id'],"status":"runner_plan_validated",
                "config_validated":True,"schedule_override_loaded":True,
                "arrivals":len(scenario.arrivals),"classes":len(scenario.label_holders),
                "raw_sample_overlap":len(scenario.duplicate_sample_ids),
                "task_examples_min":min(task_sample_counts),"task_examples_max":max(task_sample_counts),
                "class_task_occurrences_min":min(class_occurrences.values()),
                "class_task_occurrences_max":max(class_occurrences.values()),
                "outage_cohorts_match_runtime":outage_validated,
                "terminal_flush_stage_empty":not scenario.stages[-1],
                "learning_syncs":max(a.round_id for a in scenario.arrivals)+1,
                "terminal_flush":cfg.stream_terminal_flush,
                "setting":cfg.learning_setting,"model":cfg.model}
        print(json.dumps(report,sort_keys=True),flush=True)
        return 0
    from cafcl.simulator import run_experiment
    variant='taskil' if cfg.learning_setting=='task_il' else 'classil'
    runtime_root=ROOT/f'runtime_baseline_{variant}'
    run_dir=ROOT/'results'/expected['family']/expected['run_id']
    existing_entries=list(run_dir.iterdir()) if run_dir.exists() else []
    if existing_entries and not all(path.name=='worker.log' for path in existing_entries):
        status_path=run_dir/'worker_status.json'
        if not status_path.exists():
            raise RuntimeError(f'Refusing nonempty run directory without a worker status: {run_dir}')
        prior=json.loads(status_path.read_text())
        if prior.get('status') in {'completed','finished'}:
            print(json.dumps({'run_id':expected['run_id'],'status':'already_completed'},sort_keys=True),flush=True)
            return 0
        if prior.get('status')=='running':
            prior_pid=prior.get('pid')
            while True:
                try:
                    if prior_pid is None:raise ProcessLookupError
                    os.kill(int(prior_pid),0)
                except ProcessLookupError:
                    raise RuntimeError(f'Prior worker PID {prior_pid} exited without a terminal status: {run_dir}')
                except PermissionError:
                    pass
                time.sleep(WAIT_SECONDS)
                prior=json.loads(status_path.read_text())
                if prior.get('status') in {'completed','finished'}:
                    print(json.dumps({'run_id':expected['run_id'],'status':'already_completed'},sort_keys=True),flush=True)
                    return 0
                if prior.get('status')=='failed':
                    raise RuntimeError(f'Prior worker recorded failure; refusing to overwrite: {run_dir}')
        raise RuntimeError(f'Refusing to overwrite nonempty run directory: {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=True)
    config_dict=cfg.to_dict()
    write_json(run_dir/'config.json',config_dict)
    write_json(run_dir/'schedule.json',tasks)
    write_json(run_dir/'plan_validation.json',expected)
    write_json(run_dir/'source_checksum.json',source_manifest(runtime_root))
    write_json(run_dir/'design.json',{"schedule_plan":expected,"schedule":tasks,
        "config":config_dict,"source_variant":f'runtime_baseline_{variant}',
        "config_sha256":canonical_json_sha256(config_dict),
        "schedule_sha256":canonical_json_sha256(tasks)})
    write_json(run_dir/'worker_status.json',{"run_id":expected['run_id'],"status":"running","started_at":time.time(),"pid":os.getpid()})
    try:
        write_json(run_dir/'environment.json',environment_manifest(runtime_root,cfg))
        result=run_experiment(cfg,str(run_dir/'run'),str(runtime_root/'data'),download=False,verbose=True)
        write_json(run_dir/'result_summary.json',result)
        write_json(run_dir/'worker_status.json',{"run_id":expected['run_id'],"status":result.get('status','completed'),
            "finished_at":time.time(),"result":result})
        print(json.dumps({"run_id":expected['run_id'],"result":result},default=str),flush=True)
        return 0
    except BaseException as exc:
        write_json(run_dir/'worker_status.json',{"run_id":expected['run_id'],"status":"failed",
            "failed_at":time.time(),"error":repr(exc)})
        raise


def plan_only_all(plan_root):
    plan_root=Path(plan_root);results=[]
    for spec_path in sorted(plan_root.glob('[A-G]/*_seed0')):
        family=spec_path.parent.name
        setting=json.loads((spec_path/'config.json').read_text()).get('learning_setting','task_il')
        variant='taskil' if setting=='task_il' else 'classil'
        py=ROOT/f'runtime_baseline_{variant}/.venv/bin/python'
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/f'runtime_baseline_{variant}'/'src')
        env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
        p=subprocess.run([str(py),str(Path(__file__).resolve()),'--worker-spec',str(spec_path),'--plan-only'],
            cwd=ROOT/f'runtime_baseline_{variant}',env=env,text=True,capture_output=True)
        if p.returncode: raise RuntimeError(f"Runner plan validation failed: {spec_path.name}\n{p.stdout}\n{p.stderr}")
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))
    write_json(plan_root/'runner_plan_validation.json',{"status":"all_runner_plans_validated",
        "run_count":len(results),"runs":results,"updated_at":time.time()})
    print(json.dumps({"status":"all_runner_plans_validated","run_count":len(results)},indent=2))


def gpu_free_mib():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(out.strip().splitlines()[0])


def existing_campaign_jobs():
    out=subprocess.check_output(['ps','-eo','args='],text=True)
    needles=('Correlation-Aware-FCL-tasksize45-disjoint-ablation-20261122/scripts/tasksize45_campaign.py',
             'Correlation-Aware-FCL-intratask-sync-ablation-20261122/scripts/intratask_sync_campaign.py',
             'Correlation-Aware-FCL-independent-ablations-20261122/campaign.py',
             'Correlation-Aware-FCL-eight-runs-20261122/queue/',
             'FedQCL_Task/system/main.py','queue_optimizer_reset_after_order.sh','run_optimizer_reset_ablation.py')
    own=str(Path(__file__).resolve())
    return [line.strip() for line in out.splitlines() if any(n in line for n in needles) and own not in line]


def campaign(plan_root):
    if not (Path(plan_root)/'runner_plan_validation.json').exists():
        raise RuntimeError('Run --plan-all successfully before starting the queue controller')
    queue=ROOT/'queue';queue.mkdir(parents=True,exist_ok=True)
    all_runs=[]
    for family_dir in sorted(Path(plan_root).glob('[A-G]')):
        for spec_path in sorted(family_dir.glob('*_seed0')):
            expected=json.loads((spec_path/'plan_validation.json').read_text())
            all_runs.append((expected['run_id'],spec_path))
    write_json(queue/'campaign_status.json',{"status":"queued_waiting_for_capacity","queued_runs":len(all_runs),"updated_at":time.time()})
    final=[]
    for index,(run_id,spec_path) in enumerate(all_runs,1):
        status_path=ROOT/'results'/spec_path.parent.name/run_id/'worker_status.json'
        if status_path.exists():
            prior=json.loads(status_path.read_text())
            if prior.get('status') in {'completed','finished'}: final.append({"run_id":run_id,"status":"already_completed"});continue
            raise RuntimeError(f"Refusing to overwrite existing run state: {status_path}")
        while True:
            free=gpu_free_mib();external=existing_campaign_jobs()
            if free>=MIN_FREE_MIB and not external: break
            write_json(queue/'campaign_status.json',{"status":"waiting_for_capacity","next_run":run_id,
                "index":index,"runs_total":len(all_runs),"gpu_free_mib":free,
                "minimum_free_mib":MIN_FREE_MIB,"other_campaign_controllers":external,"updated_at":time.time()})
            time.sleep(WAIT_SECONDS)
        setting=json.loads((spec_path/'config.json').read_text()).get('learning_setting','task_il')
        variant='taskil' if setting=='task_il' else 'classil';py=ROOT/f'runtime_baseline_{variant}/.venv/bin/python'
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/f'runtime_baseline_{variant}'/'src')
        env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
        log=ROOT/'results'/spec_path.parent.name/run_id/'worker.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as fp:
            proc=subprocess.Popen([str(py),str(Path(__file__).resolve()),'--worker-spec',str(spec_path)],
                cwd=ROOT/f'runtime_baseline_{variant}',env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(queue/'campaign_status.json',{"status":"running_one_job","index":index,
                "runs_total":len(all_runs),"run_id":run_id,"pid":proc.pid,"started_at":time.time()})
            code=proc.wait()
        state=json.loads(status_path.read_text()) if status_path.exists() else {"status":"missing_status"}
        final.append({"run_id":run_id,"exit_code":code,"status":state.get('status')})
        write_json(queue/'campaign_status.json',{"status":"job_finished","index":index,
            "runs_total":len(all_runs),"last_run":final[-1],"updated_at":time.time()})
    write_json(queue/'campaign_status.json',{"status":"campaign_completed_or_failed_per_run","runs_total":len(all_runs),
        "results":final,"completed_at":time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--plan-all',action='store_true');p.add_argument('--plan-root',default=str(ROOT/'plans/server_compiled_v2'))
    p.add_argument('--campaign',action='store_true');a=p.parse_args()
    if a.worker_spec: return worker(a.worker_spec,a.plan_only)
    if a.plan_all: return plan_only_all(a.plan_root)
    if a.campaign:return campaign(a.plan_root)
    p.error('Choose --worker-spec, --plan-all, or --campaign')
if __name__=='__main__':main()
