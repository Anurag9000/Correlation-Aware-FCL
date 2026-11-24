"""One-command essential campaign: compile, smoke gate, run, audit, compare."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

from .runner import plan_campaign, write_json


def add_arguments(parser):
    parser.add_argument('--output',required=True,help='Campaign root (plan, results, analysis, ledger)')
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--min-free-mib',type=int,required=True)
    parser.add_argument('--min-disk-gib',type=float,default=20)
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--retries',type=int,default=0)
    parser.add_argument('--mechanistic-probes',action='store_true')
    parser.add_argument('--plan-only',action='store_true')


class Ledger:
    """Current canonical job states. Full attempt history stays in broker receipts."""
    def __init__(self,path,jobs):
        self.path=Path(path)
        self.rows={j['job_id']:{'job_id':j['job_id'],'family':j['family'],
                   'learning_setting':j['learning_setting'],'seed':j['seed'],
                   'plan_hash':j['plan_hash'],'epochs':j['epochs'],'status':'planned',
                   'result_dir':str(self.path/'results'/j['job_id'])} for j in jobs}
        self.save()

    def update(self,row):
        jid=row['job_id'];old=self.rows[jid]
        # Replace transient attempt fields; never attach a stale failure reason.
        for key in ('reason','audit','attempt','attempts','started_at','log','gpu'):
            old.pop(key,None)
        old.update(row)
        self.save()

    def save(self):
        rows=list(self.rows.values())
        write_json(self.path/'job_ledger.json',{'counts':dict(Counter(r['status'] for r in rows)),
                                              'jobs':rows})
        keys=sorted({k for r in rows for k in r})
        tmp=self.path/'job_ledger.csv.tmp'
        with tmp.open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                                 for k,v in row.items()})
        tmp.replace(self.path/'job_ledger.csv')


def run_campaign(args):
    from .broker import lease,run_broker
    from .result_audit import audit_campaign,audit_job,validate_manifest
    from .comparison import compare_campaigns
    from .essential import essential_designs
    from .curriculum import digest
    from .analysis import analyze_run
    from .broker import command_text
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=root/'plan'/'manifest.json';results=root/'results'
    with lease(root/'.essential_campaign.lock'):
        if not manifest.exists():
            plan_campaign(manifest.parent,args.data_root,0,not args.no_download,suite='essential')
        catalog=json.loads(manifest.read_text());jobs=catalog['jobs']
        expected={'J-'+d.name+'-'+setting for d in essential_designs()
                  for setting in ('task_il','class_il')}
        if (catalog.get('suite')!='essential' or len(jobs)!=60
                or {j['job_id'] for j in jobs}!=expected
                or any(j['seed']!=0 for j in jobs)
                or catalog.get('aliases') or catalog.get('infeasible')):
            raise RuntimeError('Expected the frozen 60-job seed-zero essential manifest')
        head=command_text(['git','rev-parse','HEAD'])
        if command_text(['git','status','--porcelain','--untracked-files=no']):
            raise RuntimeError('Campaign requires a clean tracked checkout')
        identity={'git_commit':head,'manifest_hash':digest(catalog),
                  'local_evaluation':True,'mechanistic_probes':args.mechanistic_probes,
                  'checkpoint_policy':'latest','client_concurrency':args.concurrency}
        identity_path=root/'campaign_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise RuntimeError('Campaign source, manifest or execution options changed; use a new root')
        write_json(identity_path,identity)
        validation=validate_manifest(manifest,args.data_root,not args.no_download)
        write_json(root/'plan_validation.json',validation)
        if validation['invalid']:raise RuntimeError('Manifest validation failed')
        ledger=Ledger(root,jobs)
        for job in jobs:
            prior=audit_job(job,results)
            if prior['status']!='not_started':ledger.update(prior)
        if args.plan_only:return {'complete':True,'planned':60,'training_launched':False}
        options=vars(args).copy()
        options.update(manifest=str(manifest),output=str(results),limit=0,
                       local_evaluation=True,checkpoint_policy='latest',
                       allow_shared_gpu=False,no_download=True)
        smoke=run_broker(SimpleNamespace(**dict(options,family='smoke',exclude_family=None)),
                         progress_callback=ledger.update)
        phases={'smoke':smoke}
        # No production jobs until all eight smoke conditions independently pass.
        if smoke['verified']==8 and not smoke['failed'] and not smoke['blocked']:
            phases['main']=run_broker(SimpleNamespace(**dict(options,family=None,exclude_family='smoke')),
                                      progress_callback=ledger.update)
        audit=audit_campaign(manifest,results)
        for row in audit['jobs']:
            if row['status']=='not_started' and ledger.rows[row['job_id']]['status'] in ('blocked','failed'):
                continue
            ledger.update(row)
        analysis_errors=[]
        for row in audit['jobs']:
            if row['status']=='artifact_verified':
                try:analyze_run(results/row['job_id'],figures=True)
                except Exception as exc:analysis_errors.append({'job_id':row['job_id'],'reason':repr(exc)})
        comparison=compare_campaigns([(str(manifest),str(results))],root/'comparison')
        complete=(audit['counts'].get('artifact_verified',0)==60
                  and not analysis_errors
                  and not comparison['condition_status_counts'].get('analysis_failed',0)
                  and all(not p['failed'] and not p['blocked'] for p in phases.values()))
        summary={'complete':complete,'planned':60,'phases':phases,
                 'audit_counts':audit['counts'],'analysis_errors':analysis_errors,
                 'comparison':comparison}
        write_json(root/'campaign_summary.json',summary)
        return summary
