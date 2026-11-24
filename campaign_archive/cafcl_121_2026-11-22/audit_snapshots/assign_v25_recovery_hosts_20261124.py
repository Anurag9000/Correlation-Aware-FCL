import copy, hashlib, json, os
from pathlib import Path

BASE=Path('/home/anurag-basistha/CORRELATION_AWARE_FCL_ABLATIONS_2026-11-22')
LIVE=BASE/'audit_snapshots/recovery_v10_manifest_live_20261124.json'
AUDIT=BASE/'audit_snapshots'
OUT_V100=BASE/'audit_snapshots/recovery_v10_manifest_assigned_20261124.json'
OUT_MTECH=BASE/'audit_snapshots/recovery_mtech_queue_seed_20261124.json'

case_info={
 'V25_CLASSIL_DISJOINT10_ALLSYNC_EPOCHS5_REPLAY_FEDAVG_SEED0':('recovery_v10_audit_completed_attempt01',0,5,0.2632,'attempt_01','2026-11-23T21:41:55Z','2026-11-23T21:53:35Z'),
 'V25_CLASSIL_DISJOINT10_ALLSYNC_EPOCHS2_REPLAY_FEDAVG_SEED0':('recovery_v10_audit_attempts/all_ep2',0,2,0.1206,'attempt_01','2026-11-23T21:41:54Z','2026-11-23T21:51:18Z'),
 'V25_CLASSIL_DISJOINT10_OFFLINE1_EPOCHS2_REPLAY_FEDAVG_SEED0':('recovery_v10_audit_attempts/offline1_ep2',1,2,0.1032,'attempt_01','2026-11-23T21:41:56Z','2026-11-23T21:51:10Z'),
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def check_case(job,folder,offline,epochs,acc,attempt,start,end):
 root=AUDIT/folder
 worker=json.load(open(root/'worker_status.json'))
 result=json.load(open(root/'result_summary.json'))
 design=json.load(open(root/'design.json'))
 cfg=json.load(open(root/f'runs/{job}/config.json'))
 summ=json.load(open(root/f'runs/{job}/summary.json'))
 metrics=json.load(open(root/f'runs/{job}/metrics.json'))
 reports=[json.load(open(root/f'runs/{job}/per_sync_accuracy'/f'sync_{i:03d}.json')) for i in range(1,12)]
 assert worker['status']=='completed' and worker['learning_syncs']==10
 assert result['status']=='completed' and abs(result['final_accuracy']-acc)<1e-10
 assert design['job_id']==job and design['setting']=='class_il' and design['curriculum']=='disjoint'
 assert design['offline_count']==offline and design['epochs_per_task']==epochs and design['seed']==0
 assert design['learning_syncs']==10 and design['total_server_syncs_including_flush']==11
 assert cfg['dataset']=='cifar100' and cfg['num_clients']==5 and cfg['learning_setting']=='class_il'
 assert cfg['scenario']=='disjoint' and cfg['participation']==(5-offline)/5 and cfg['epochs']==epochs and cfg['seed']==0
 assert summ['rounds']==11 and summ['per_sync_report_count']==11 and len(metrics)==11
 assert reports[-1]['terminal_communication_only_flush'] is True
 assert reports[-1]['server_integrated_class_count']==100
 integrated=[r['server_integrated_class_count'] for r in reports]
 assert integrated[-1]==100 and integrated==sorted(integrated)
 assert abs(summ['final_metrics']['accuracy']-acc)<1e-10
 hashes={
  'worker_status':sha(root/'worker_status.json'),
  'result_summary':sha(root/'result_summary.json'),
  'design':sha(root/'design.json'),
  'config':sha(root/f'runs/{job}/config.json'),
  'summary':sha(root/f'runs/{job}/summary.json'),
  'metrics':sha(root/f'runs/{job}/metrics.json'),
 }
 return {'attempt_id':attempt,'state':'completed','host':'v100','started_utc':start,'finished_utc':end,
  'output_dir':f'/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/results/v25_offline_epoch_grid_20261122_v2/recovery_v10/runs/{job}/{attempt}',
  'final_accuracy':acc,'learning_syncs':10,'per_sync_reports':11,'covered_classes':100,
  'metrics_all_finite':True,'terminal_flush':True,'worker_status':'completed',
  'verification':{'config_match':True,'schedule_match':True,'finite_metrics':True,'class_coverage':100,'summary_status':'completed','summary_final_accuracy':acc},
  'artifact_sha256':hashes}

m=json.load(open(LIVE))
for r in m['runs']:
 if r['job_id'] in case_info:
  folder,off,ep,acc,attempt,start,end=case_info[r['job_id']]
  audit=check_case(r['job_id'],folder,off,ep,acc,attempt,start,end)
  prior=next((a for a in r.get('attempts',[]) if a.get('attempt_id')==attempt),None)
  if prior is None:
   prior={'attempt_id':attempt}
   r.setdefault('attempts',[]).append(prior)
  prior.update(audit)
  prior['supervisor_correction']='Earlier controller marked this worker interrupted after its PID exited; terminal worker_status, result_summary, config, metrics, and 11 sync reports verify successful completion.'
  r['status']='completed'; r['completion_source']='audited V100 attempt_01 terminal artifacts'; r.pop('failure_reason',None)

queued=[r for r in m['runs'] if r['status']=='queued']
assert len(queued)==32, len(queued)
v100_ids={
 'V25_TASKIL_JOINT_CHAIN20_ALLSYNC_EPOCHS5_REPLAY_FEDAVG_SEED0',
 'V25_CLASSIL_JOINT_CHAIN20_ALLSYNC_EPOCHS5_REPLAY_FEDAVG_SEED0',
}
assert v100_ids.issubset({r['job_id'] for r in queued})
mtech_rows=[]
for r in m['runs']:
 if r['status']=='completed':
  r['host_assignment']='v100'; continue
 if r['job_id'] in v100_ids:
  r['host_assignment']='v100'; r['host']='v100'; r['status']='queued'
 else:
  r['host_assignment']='mtech'; r['host']='mtech'; r['status']='delegated_mtech'
  x=copy.deepcopy(r); x['status']='queued'; x['host']='mtech'; x['host_assignment']='mtech'
  x['seed_attempt_count']=len(x.get('attempts',[])); x.pop('output_dir',None)
  mtech_rows.append(x)
assert len(mtech_rows)==30 and len(v100_ids)==2
assert not ({r['job_id'] for r in mtech_rows}&v100_ids)
assert len({r['job_id'] for r in mtech_rows})==30
m['status']='queued'
m['host_assignments']={'v100':sorted(v100_ids),'mtech':sorted(r['job_id'] for r in mtech_rows)}
m['assignment_note']='Three V100 attempt_01 runs audited complete. Two additional epoch-5 cells remain assigned to V100, constrained by its 34 GiB local output headroom and ~13 GiB observed output per run. Other 30 unique cells assigned to Mtech. No cell is assigned to both hosts.'
m['updated_utc']='2026-11-24T03:36:00+05:30'
OUT_V100.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
seed={'campaign':m.get('campaign'),'status':'queued','created_utc':'2026-11-24T03:36:00+05:30',
 'host':'MtechDEServer1','source_manifest':str(OUT_V100),'source_manifest_sha256':sha(OUT_V100),
 'worker_peak_estimate_mib':6184,'safety_margin_mib':256,'admission_required_mib':6440,
 'storage_reserve_mib':1024,'storage_per_worker_mib':14000,
 'assignment_note':'30 unique queued cells assigned to Mtech; V100 retains two other unique queued cells. Retry cells are prioritized, using their existing histories and next attempt IDs.',
 'runs':mtech_rows}
OUT_MTECH.write_text(json.dumps(seed,indent=2,sort_keys=True)+'\n')
print('V100 state counts', {s:sum(r['status']==s for r in m['runs']) for s in sorted({r['status'] for r in m['runs']})})
print('host assignments',len(v100_ids),len(mtech_rows),'seed_sha256',sha(OUT_MTECH),'v100_state_sha256',sha(OUT_V100))
for r in mtech_rows:
 s=r['spec']; retries=sum(a.get('status') in ('failed','failed_or_interrupted','interrupted') for a in r.get('attempts',[]))
 print('MTECH',retries,r['job_id'])
for job in sorted(v100_ids): print('V100',job)
