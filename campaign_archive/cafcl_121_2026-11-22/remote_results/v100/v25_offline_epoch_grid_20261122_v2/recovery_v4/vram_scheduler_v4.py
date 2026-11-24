#!/usr/bin/env python3
"""VRAM-admitted, append-only continuation for the 48-cell V100 grid.

Runs the documented worker from its existing V100 checkout, storing all new
attempts, logs and controller state on the Kakarot-backed result mount.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path('/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122')
CAMPAIGN = REPO / 'results/v25_offline_epoch_grid_20261122_v2'
ORIGINAL = CAMPAIGN / 'manifest.json'
OLD_OVERRIDE = CAMPAIGN / 'sequential_override/manifest.json'
WORKER = REPO / 'scripts/run_v25_offline_epoch_grid.py'
PYTHON = REPO / '.venv/bin/python'
SOURCE_ROOT = REPO / 'source_variants'
ROOT = Path(os.environ.get('CAFCL_OUTPUT_ROOT',
    '/dev/shm/cafcl-kakarot-results/v100/v25_offline_epoch_grid_20261122_v2/recovery'))
TMPROOT = Path(os.environ.get('CAFCL_TMP_DIR','/dev/shm/cafcl-tmp'))
RUNS = ROOT / 'runs'
STATE = ROOT / 'recovery_manifest.json'
LOGDIR = ROOT / 'logs'
PEAK_MIB = 9000  # observed single-worker V100 delta was about 8.5 GiB
MARGIN_MIB = 1024
POLL_SEC = 20
MAX_RETRIES = 100  # continue retries while failures are diagnosed and repaired


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def gpu_free():
    out = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
    vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
    if len(vals) != 1: raise RuntimeError('expected one V100 free-memory reading, got '+repr(vals))
    return vals[0]


def gpu_by_pid():
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL)
    ans={}
    for line in out.splitlines():
        try:
            pid, mem = [int(x.strip()) for x in line.split(',',1)]
            ans[pid]=mem
        except Exception: pass
    return ans


def descendants(pid):
    rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()
    kids={}
    for row in rows:
        try:
            a,b=map(int,row.split()); kids.setdefault(b,[]).append(a)
        except Exception: pass
    out={pid}; todo=[pid]
    while todo:
        p=todo.pop()
        for c in kids.get(p,[]):
            if c not in out: out.add(c); todo.append(c)
    return out


def alive(pid):
    try:
        pid=int(pid); os.kill(pid,0)
        state=subprocess.check_output(['ps','-o','stat=','-p',str(pid)],text=True).strip()
        return bool(state) and not state.startswith('Z')
    except (ProcessLookupError,PermissionError,ValueError): return False


def verify_done(out: Path):
    try:
        s=json.loads((out/'worker_status.json').read_text())
        return s.get('status')=='completed' and (out/'result_summary.json').is_file()
    except Exception: return False


def prepare():
    original=json.loads(ORIGINAL.read_text())
    old=json.loads(OLD_OVERRIDE.read_text())
    if len(original.get('runs',[])) != 48 or len(old.get('runs',[])) != 48:
        raise RuntimeError('V100 manifests do not each contain 48 runs')
    # Refuse launch while any old campaign worker is still alive.
    for row in old['runs']:
        if row.get('status')=='running' and row.get('pid') and alive(row['pid']):
            raise RuntimeError('Existing V100 campaign worker is alive: '+str(row['pid']))
    original_by={r['job_id']:r for r in original['runs']}
    state={'campaign':original.get('campaign'),'source_manifest':str(ORIGINAL),
           'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'status':'running','worker_peak_estimate_mib':PEAK_MIB,
           'safety_margin_mib':MARGIN_MIB,'admission_required_mib':PEAK_MIB+MARGIN_MIB,
           'attempts':[],'runs':[]}
    for row in old['runs']:
        job=row['job_id']; prev=Path(row['output_dir'])
        if row.get('status')=='completed' and verify_done(prev):
            state['runs'].append({'job_id':job,'status':'completed','completion_source':'existing verified worker status and result summary','output_dir':str(prev),'attempts':[]})
            continue
        if row.get('status')=='running':
            # Process absent is the authoritative stale-state check; preserve old files untouched.
            pid=row.get('pid')
            if pid and alive(pid): raise RuntimeError('Refusing retry; old worker PID alive: '+str(pid))
            try:
                old_status=json.loads((prev/'worker_status.json').read_text())
                worker_pid=old_status.get('pid')
                if old_status.get('status')=='running' and worker_pid and alive(worker_pid):
                    raise RuntimeError('Refusing retry; original worker PID is still alive: '+str(worker_pid))
            except FileNotFoundError:
                pass
            failed={'attempt_id':'original_stale','status':'failed_or_interrupted','output_dir':str(prev),'pid':pid,'reason':'manifest said running but PID is absent; original files/logs retained'}
        else: failed=None
        pending={'job_id':job,'status':'queued','attempts':[]}
        if failed: pending['attempts'].append(failed)
        # Pull exact immutable design factors from the original manifest.
        spec=original_by[job]
        pending['spec']=spec
        state['runs'].append(pending)
    ROOT.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True); LOGDIR.mkdir(parents=True,exist_ok=True)
    atomic(STATE,state)
    return state


def launch(row, active, admission_free):
    spec=row['spec']; job=row['job_id']
    n=sum(1 for a in row['attempts'] if a.get('status') in ('failed','failed_or_interrupted','interrupted'))
    attempt='attempt_{:02d}'.format(n+1)
    out=RUNS/job/attempt; log=LOGDIR/(job+'_'+attempt+'.log')
    setting=spec['setting']; source=SOURCE_ROOT/('taskil_src' if setting=='task_il' else 'classil_src')
    env=os.environ.copy(); env.update({'PYTHONPATH':str(source/'src'),'CAFCL_SOURCE_DIR':str(source),
      'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
      'NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(TMPROOT),
      'XDG_CACHE_HOME':str(TMPROOT/'xdg'),'MPLCONFIGDIR':str(TMPROOT/'matplotlib'),
      'TORCH_HOME':str(TMPROOT/'torch')})
    cmd=[str(PYTHON),str(WORKER),'--worker','--setting',setting,'--curriculum',spec['curriculum'],
         '--offline-count',str(spec['offline_count']),'--epochs',str(spec['epochs_per_task']),
         '--job-id',job,'--block-id',spec['block_id'],'--output-dir',str(out)]
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as f:
        p=subprocess.Popen(cmd,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rec={'attempt_id':attempt,'status':'running','pid':p.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'output_dir':str(out),'log':str(log),'command':cmd,'prelaunch_free_vram_mib':admission_free,
         'peak_estimate_mib':PEAK_MIB,'safety_margin_mib':MARGIN_MIB}
    row['attempts'].append(rec); row['status']='running'; row['active_attempt']=attempt
    active[p.pid]=(p,row,rec)
    return p.pid


def main():
    for d in (TMPROOT,TMPROOT/'xdg',TMPROOT/'matplotlib',TMPROOT/'torch'):
        d.mkdir(parents=True,exist_ok=True)
    if STATE.exists():
        state=json.loads(STATE.read_text())
        if state.get('status')=='completed': print('recovery already completed',flush=True); return 0
        changed=False
        for row in state.get('runs',[]):
            for rec in row.get('attempts',[]):
                log=rec.get('log')
                if rec.get('status')=='failed' and log:
                    try:
                        if 'No usable temporary directory found' in Path(log).read_text(errors='replace'):
                            rec['failure_class']='environment_tempdir'
                            rec['failure_reason']='worker could not find writable temp directory; retry uses /dev/shm/cafcl-tmp'
                            changed=True
                    except OSError: pass
            if row.get('status')=='running':
                rec=next((a for a in reversed(row.get('attempts',[])) if a.get('status')=='running'),None)
                if rec and rec.get('pid') and alive(rec['pid']):
                    raise RuntimeError('Recovery worker still alive; refusing duplicate controller: '+str(rec['pid']))
                if rec:
                    rec['status']='failed'; rec['failure_class']='controller_interruption'
                    rec['failure_reason']='worker PID absent during recovery; previous attempt outputs and logs retained for audit'
                    rec['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
                row['status']='queued'; row.pop('active_attempt',None); changed=True
        if changed: atomic(STATE,state)
    else: state=prepare()
    active={}
    while True:
        # Reconcile exits and verify trained artifacts before marking completed.
        for pid,(proc,row,rec) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            rec['return_code']=code; rec['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            if code==0 and verify_done(Path(rec['output_dir'])):
                rec['status']='completed'; row['status']='completed'; row.pop('active_attempt',None)
            else:
                rec['status']='failed'; rec['failure_reason']='worker exited {} or terminal status/summary missing'.format(code)
                row['status']='queued'; row.pop('active_attempt',None)
            del active[pid]; atomic(STATE,state)
        pending=[r for r in state['runs'] if r['status']=='queued']
        if not pending and not active:
            state['status']='completed' if all(r['status']=='completed' for r in state['runs']) else 'needs_repair'
            state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()); atomic(STATE,state)
            return 0 if state['status']=='completed' else 2
        # Account for workers whose expected peak is not yet reflected in measured free VRAM.
        free=gpu_free(); mem=gpu_by_pid(); outstanding=0
        for pid,(_,_,rec) in active.items():
            observed=sum(mem.get(q,0) for q in descendants(pid))
            outstanding+=max(0,PEAK_MIB-observed)
        admitted=0
        while pending and free-outstanding >= PEAK_MIB+MARGIN_MIB:
            row=pending.pop(0)
            nfailed=sum(1 for a in row['attempts'] if a.get('status')=='failed' and not a.get('failure_class'))
            if nfailed>=MAX_RETRIES:
                row['status']='needs_repair'; state['status']='needs_repair'; atomic(STATE,state); continue
            pid=launch(row,active,free); admitted+=1
            # Reserve immediately even before the CUDA context appears.
            outstanding+=PEAK_MIB
            state['last_admission']={'job_id':row['job_id'],'pid':pid,'free_vram_mib':free,
                'active_workers':len(active),'admitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            atomic(STATE,state)
            free=gpu_free(); mem=gpu_by_pid()
            observed=sum(mem.get(q,0) for q in descendants(pid))
            outstanding=max(0,outstanding-PEAK_MIB)+max(0,PEAK_MIB-observed)
        state['last_resource_sample']={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'free_vram_mib':free,'active_worker_pids':list(active),'outstanding_reserved_mib':outstanding,
           'launches_this_pass':admitted}
        atomic(STATE,state)
        time.sleep(POLL_SEC)

if __name__=='__main__':
    raise SystemExit(main())
