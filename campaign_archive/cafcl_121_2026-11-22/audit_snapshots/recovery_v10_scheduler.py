#!/usr/bin/env python3
"""VRAM and disk admitted resume for the V100 epoch-participation grid."""
from __future__ import annotations
import copy, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path('/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122')
WORKER = REPO / 'scripts/run_v25_offline_epoch_grid.py'
PYTHON = REPO / '.venv/bin/python'
SEED = Path('/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/results/v25_offline_epoch_grid_20261122_v2/recovery_v10_seed_manifest_20261124.json')
ROOT = Path('/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/results/v25_offline_epoch_grid_20261122_v2/recovery_v10')
TMPROOT = Path('/dev/shm/cafcl-tmp')
RCLONE = '/dev/shm/cafcl-tools/rclone'
RCLONE_CONFIG = '/dev/shm/cafcl-tools/rclone.conf'
REMOTE_ROOT = 'kakarot:v100/v25_offline_epoch_grid_20261122_v2/recovery_v10'
PEAK_MIB = 6184  # prior per-worker peak observed on this V100 worker
MARGIN_MIB = 256
STORAGE_RESERVE_MIB = 1024
STORAGE_PER_WORKER_MIB = 3500  # prior measured checkpoint/cache budget
POLL_SEC = 20
MAX_RETRIES = 8
STATE = ROOT / 'recovery_manifest.json'
LOGDIR = ROOT / 'logs'
RUNS = ROOT / 'runs'
PREFLIGHT = ROOT / 'preflight'


def atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def gpu_free():
    out = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
    vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
    if len(vals) != 1:
        raise RuntimeError('expected exactly one V100 free-VRAM reading, got '+repr(vals))
    return vals[0]


def gpu_by_pid():
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL)
    ans = {}
    for line in out.splitlines():
        try:
            pid, mem = [int(x.strip()) for x in line.split(',',1)]
            ans[pid] = mem
        except Exception:
            pass
    return ans


def descendants(pid):
    rows = subprocess.check_output(['ps','-eo','pid=,ppid='], text=True).splitlines()
    kids = {}
    for row in rows:
        try:
            a,b = map(int,row.split()); kids.setdefault(b,[]).append(a)
        except Exception:
            pass
    out, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        for c in kids.get(p,[]):
            if c not in out:
                out.add(c); todo.append(c)
    return out


def alive(pid):
    try:
        os.kill(int(pid),0)
        state = subprocess.check_output(['ps','-o','stat=','-p',str(pid)], text=True).strip()
        return bool(state) and not state.startswith('Z')
    except (ProcessLookupError,PermissionError,ValueError):
        return False


def verify_done(out: Path):
    try:
        s = json.loads((out/'worker_status.json').read_text())
        summary = json.loads((out/'result_summary.json').read_text())
        return s.get('status') == 'completed' and isinstance(summary,dict)
    except Exception:
        return False


def run_cmd(cmd, timeout=None, env=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout, env=env)


def publish(path: Path, remote: str):
    cmd = [RCLONE, '--config', RCLONE_CONFIG, 'copyto', str(path), remote]
    res = run_cmd(cmd, timeout=180)
    if res.returncode:
        raise RuntimeError('rclone copyto failed: '+res.stdout[-2000:])


def publish_tree(path: Path, remote: str):
    cmd = [RCLONE, '--config', RCLONE_CONFIG, 'copy', str(path), remote]
    res = run_cmd(cmd, timeout=600)
    if res.returncode:
        raise RuntimeError('rclone copy failed: '+res.stdout[-2000:])


def save(state, publish_remote=True):
    state['updated_utc'] = utc()
    atomic(STATE,state)
    if publish_remote:
        try:
            publish(STATE, REMOTE_ROOT+'/recovery_manifest.json')
            state['remote_manifest_synced_utc'] = utc()
            atomic(STATE,state)
        except Exception as exc:
            state.setdefault('archive_errors',[]).append({'at':utc(),'error':repr(exc)})
            atomic(STATE,state)


def seed_state():
    raw = SEED.read_bytes()
    seed = json.loads(raw)
    rows = seed.get('runs',[])
    ids = [r.get('job_id') for r in rows]
    if len(rows) != 48 or len(set(ids)) != 48:
        raise RuntimeError('recovery seed must have exactly 48 unique cells')
    if sum(r.get('status') == 'completed' for r in rows) != 13:
        raise RuntimeError('seed completion count changed; stop for reconciliation')
    if sum(r.get('status') == 'queued' for r in rows) != 35:
        raise RuntimeError('seed queue count changed; stop for reconciliation')
    out = {
        'campaign': seed.get('campaign'), 'status':'preflight',
        'source_seed': str(SEED), 'source_seed_sha256':hashlib.sha256(raw).hexdigest(),
        'created_utc':utc(), 'host':'V100-server',
        'worker_peak_estimate_mib':PEAK_MIB, 'safety_margin_mib':MARGIN_MIB,
        'admission_required_mib':PEAK_MIB+MARGIN_MIB,
        'storage_reserve_mib':STORAGE_RESERVE_MIB,
        'storage_per_worker_mib':STORAGE_PER_WORKER_MIB,
        'runs':[]
    }
    for old in rows:
        r = copy.deepcopy(old)
        r['attempts'] = r.get('attempts',[])
        r['seed_attempt_count'] = len(r['attempts'])
        r['status'] = 'completed' if old.get('status') == 'completed' else 'queued'
        r.pop('output_dir',None) if r['status'] == 'queued' else None
        out['runs'].append(r)
    ROOT.mkdir(parents=True,exist_ok=False)
    RUNS.mkdir(); LOGDIR.mkdir(); PREFLIGHT.mkdir()
    atomic(STATE,out)
    return out


def env_for(spec):
    source = REPO/'source_variants'/('taskil_src' if spec['setting']=='task_il' else 'classil_src')
    env = os.environ.copy()
    env.update({
        'PYTHONPATH':str(source/'src'), 'CAFCL_SOURCE_DIR':str(source),
        'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(TMPROOT),
        'TEMP':str(TMPROOT),'TMP':str(TMPROOT),'HOME':str(TMPROOT/'home'),
        'XDG_CACHE_HOME':str(TMPROOT/'cache'),
        'TORCHINDUCTOR_CACHE_DIR':str(TMPROOT/'cache'/'inductor'),
        'MPLCONFIGDIR':str(TMPROOT/'matplotlib'),'TORCH_HOME':str(TMPROOT/'torch'),
        # The controller owns measured VRAM reservations. Bypass the old hard-coded 12 GiB wrapper.
        'CAFCL_VRAM_BROKER_ADMITTED':'1',
    })
    return env,source


def worker_command(spec, out):
    return [str(PYTHON),str(WORKER),'--worker','--setting',spec['setting'],
            '--curriculum',spec['curriculum'],'--offline-count',str(spec['offline_count']),
            '--epochs',str(spec['epochs_per_task']),'--job-id',spec['job_id'],
            '--block-id',spec['block_id'],'--output-dir',str(out)]


def preflight(state):
    failures=[]
    for row in state['runs']:
        if row['status']=='completed':
            continue
        if row.get('preflight',{}).get('return_code') == 0:
            continue
        spec=row['spec']; out=PREFLIGHT/row['job_id']
        env,_=env_for(spec); cmd=worker_command(spec,out)+['--plan-only']
        log=LOGDIR/(row['job_id']+'_preflight.log')
        res=run_cmd(cmd,env=env)
        log.write_text(res.stdout)
        evidence={'return_code':res.returncode,'log':str(log),'completed_utc':utc()}
        if res.returncode==0:
            try:
                design=json.loads((out/'design.json').read_text())
                evidence.update({'learning_syncs':design['learning_syncs'],
                    'terminal_flush':design['terminal_flush'],
                    'task_instances':design['task_instances'],
                    'class_occurrence_values':design['class_occurrence_values']})
                if design['learning_syncs'] != (20 if spec['curriculum']=='joint' else 10):
                    raise ValueError('unexpected learning sync count')
                row['preflight']=evidence
            except Exception as exc:
                evidence['error']=repr(exc); failures.append(row['job_id']); row['preflight']=evidence
        else:
            evidence['tail']=res.stdout[-3000:]; row['preflight']=evidence; failures.append(row['job_id'])
        atomic(STATE,state)
    if failures:
        state['status']='plan_validation_failed'; state['plan_validation_failures']=failures; save(state)
        raise RuntimeError('plan validation failed for '+', '.join(failures))
    state['preflight_completed_utc']=utc(); state['status']='queued'; save(state)


def next_attempt(row):
    nums=[]
    for a in row.get('attempts',[]):
        x=str(a.get('attempt_id',''))
        if x.startswith('attempt_'):
            try: nums.append(int(x.split('_',1)[1]))
            except ValueError: pass
    return 'attempt_{:02d}'.format(max(nums,default=0)+1)


def launch(row, active, free):
    spec=row['spec']; job=row['job_id']; attempt=next_attempt(row)
    out=RUNS/job/attempt; log=LOGDIR/(job+'_'+attempt+'.log')
    env,_=env_for(spec); cmd=worker_command(spec,out)
    out.mkdir(parents=True,exist_ok=False); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as f:
        proc=subprocess.Popen(cmd,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rec={'attempt_id':attempt,'status':'running','pid':proc.pid,'started_utc':utc(),
         'output_dir':str(out),'log':str(log),'command':cmd,
         'prelaunch_free_vram_mib':free,'peak_estimate_mib':PEAK_MIB,
         'safety_margin_mib':MARGIN_MIB,'source_dir':str((REPO/'source_variants'/('taskil_src' if spec['setting']=='task_il' else 'classil_src')))}
    row.setdefault('attempts',[]).append(rec); row['status']='running'; row['active_attempt_id']=attempt
    active[proc.pid]=(proc,row,rec)
    save(ACTIVE_STATE[0])
    return proc.pid


ACTIVE_STATE=[None]


def reconcile(active,state):
    changed=False
    for pid,(proc,row,rec) in list(active.items()):
        code=proc.poll()
        if code is None: continue
        rec['return_code']=code if isinstance(proc,subprocess.Popen) else None
        rec['finished_utc']=utc()
        out=Path(rec['output_dir'])
        if code==0 and verify_done(out):
            rec['status']='completed'; row['status']='completed'; row.pop('active_attempt_id',None)
            rec['verification']={'worker_status':'completed','result_summary_present':True}
            try:
                publish_tree(out,REMOTE_ROOT+'/runs/'+row['job_id']+'/'+rec['attempt_id'])
                publish(Path(rec['log']),REMOTE_ROOT+'/logs/'+Path(rec['log']).name)
                rec['archive_status']='copied_to_b550_webdav'; rec['archive_completed_utc']=utc()
            except Exception as exc:
                rec['archive_status']='pending'; rec['archive_error']=repr(exc)
        else:
            rec['status']='failed'; rec['failure_reason']='worker exit {} or terminal status/result summary missing'.format(code)
            row['status']='queued'; row.pop('active_attempt_id',None)
        del active[pid]; changed=True
    if changed: save(state)


def active_outstanding(active):
    mem=gpu_by_pid(); outstanding=0
    for pid,(_,_,_) in active.items():
        observed=sum(mem.get(q,0) for q in descendants(pid))
        outstanding+=max(0,PEAK_MIB-observed)
    return outstanding


class DetachedProcess:
    """Poll a worker adopted after its controller was interrupted."""
    def __init__(self,pid,output):
        self.pid=int(pid); self.output=Path(output)
    def poll(self):
        if alive(self.pid): return None
        return 0 if verify_done(self.output) else 1


def main():
    for d in (TMPROOT,TMPROOT/'home',TMPROOT/'cache',TMPROOT/'cache'/'inductor',TMPROOT/'matplotlib',TMPROOT/'torch'):
        d.mkdir(parents=True,exist_ok=True)
    if STATE.exists():
        state=json.loads(STATE.read_text())
        if state.get('status') in ('completed','plan_validation_failed'):
            raise RuntimeError('recovery state already terminal; refusing blind resume')
        baseline={r['job_id']:len(r.get('attempts',[])) for r in json.loads(SEED.read_text()).get('runs',[])}
        for row in state['runs']:
            row.setdefault('seed_attempt_count',baseline.get(row['job_id'],len(row.get('attempts',[]))))
            if row.get('status')=='needs_repair' and row.get('failure_reason')=='retry limit reached; preserve attempts and diagnose':
                row['status']='queued'
                row['supervisor_recovery_note']='Requeued: prior controller counted pre-v10 historical failures against the new retry limit.'
                row.pop('failure_reason',None)
            if row.get('status')=='running':
                rec=next((a for a in reversed(row.get('attempts',[])) if a.get('status')=='running'),None)
                if rec and rec.get('pid') and alive(rec['pid']):
                    continue
                if rec:
                    rec['status']='failed_or_interrupted'; rec['failure_reason']='controller restart found worker PID absent; attempt files retained'
                    rec['finished_utc']=utc()
                row['status']='queued'; row.pop('active_attempt_id',None)
        if any(r['status']=='queued' and r.get('preflight',{}).get('return_code') != 0 for r in state['runs']):
            state['status']='preflight'; preflight(state)
        save(state)
    else:
        state=seed_state()
        # The seed and plan source are archived before workers can start.
        publish(SEED,REMOTE_ROOT+'/recovery_v10_seed_manifest_20261124.json')
        preflight(state)
    ACTIVE_STATE[0]=state
    active={}
    # Adopt live detached workers instead of duplicating or interrupting them.
    for row in state['runs']:
        if row.get('status')!='running': continue
        rec=next((a for a in reversed(row.get('attempts',[])) if a.get('status')=='running'),None)
        if rec and rec.get('pid') and alive(rec['pid']):
            active[int(rec['pid'])]=(DetachedProcess(rec['pid'],rec['output_dir']),row,rec)
        else:
            row['status']='queued'; row.pop('active_attempt_id',None)
    while True:
        reconcile(active,state)
        pending=[r for r in state['runs'] if r['status']=='queued']
        if not pending and not active:
            state['status']='completed' if all(r['status']=='completed' for r in state['runs']) else 'needs_repair'
            state['finished_utc']=utc(); save(state); return 0 if state['status']=='completed' else 2
        # Put any cell with prior failed/interrupted attempts ahead of never-started cells.
        pending.sort(key=lambda r:(-sum(a.get('status') in ('failed','failed_or_interrupted','interrupted') for a in r.get('attempts',[])), r['job_id']))
        free=gpu_free(); outstanding=active_outstanding(active)
        disk_free=shutil.disk_usage(ROOT).free//(1024*1024)
        storage_outstanding=len(active)*STORAGE_PER_WORKER_MIB
        launched=[]
        while pending and free-outstanding >= PEAK_MIB+MARGIN_MIB and disk_free-storage_outstanding >= STORAGE_RESERVE_MIB+STORAGE_PER_WORKER_MIB:
            row=pending.pop(0)
            baseline=int(row.get('seed_attempt_count',0))
            failed=sum(a.get('status') in ('failed','failed_or_interrupted','interrupted')
                       for a in row.get('attempts',[])[baseline:])
            if failed>=MAX_RETRIES:
                row['status']='needs_repair'; row['failure_reason']='retry limit reached; preserve attempts and diagnose'; save(state); continue
            pid=launch(row,active,free); launched.append({'job_id':row['job_id'],'pid':pid,'attempt':row['active_attempt_id']})
            storage_outstanding+=STORAGE_PER_WORKER_MIB
            time.sleep(1)
            free=gpu_free(); outstanding=active_outstanding(active)
            disk_free=shutil.disk_usage(ROOT).free//(1024*1024)
        state['last_resource_sample']={'utc':utc(),'free_vram_mib':free,
            'active_worker_pids':list(active),'outstanding_vram_reservation_mib':outstanding,
            'worker_peak_estimate_mib':PEAK_MIB,'margin_mib':MARGIN_MIB,
            'disk_free_mib':disk_free,'storage_outstanding_reservation_mib':storage_outstanding,
            'launches_this_pass':launched}
        save(state)
        time.sleep(POLL_SEC)


if __name__=='__main__':
    raise SystemExit(main())
