"""Plan revalidation and read-only terminal campaign audit."""
import csv
import json
from collections import Counter
from pathlib import Path
from .curriculum import digest


def validate_manifest(manifest,data_root,download=False):
    from .adapter import load_plan
    from ..data import get_data_bank
    path=Path(manifest).resolve()
    jobs=json.loads(path.read_text())['jobs'];banks={};invalid=[];valid=0
    for job in jobs:
        try:
            seed=job['seed']
            if seed not in banks:
                banks[seed]=get_data_bank('cifar100',data_root,download=download,seed=seed)
            load_plan(path.parent/job['plan_file'],job['plan_hash'],banks[seed].train.targets)
            valid+=1
        except Exception as exc:invalid.append({'job_id':job['job_id'],'reason':repr(exc)})
    return {'valid':valid,'invalid':invalid}


def audit_campaign(manifest,output):
    from ..stage_report import assert_sync_reports_complete
    from .runner import write_json
    jobs=json.loads(Path(manifest).read_text())['jobs']
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for job in jobs:rows.append(audit_job(job,root))
    report={'counts':dict(Counter(x['status'] for x in rows)),'jobs':rows,
            'scope':'Artifact completeness, not causal validity or independent checkpoint inference'}
    write_json(root/'joint_campaign_audit.json',report)
    keys=sorted({k for row in rows for k in row})
    with (root/'joint_campaign_audit.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return report


def audit_job(job,root):
    from ..stage_report import assert_sync_reports_complete
    root=Path(root)
    directory=root/job['job_id'];row={'job_id':job['job_id'],'status':'not_started'}
    try:
        status=directory/'status.json'
        if status.exists():row['status']=json.loads(status.read_text())['status']
        if row['status']=='completed':
            plan=json.loads((directory/'joint_plan.json').read_text())
            if plan['plan_hash']!=job['plan_hash'] or digest({k:v for k,v in plan.items() if k!='plan_hash'})!=job['plan_hash']:
                raise ValueError('Run plan hash mismatch')
            config=json.loads((directory/'config.json').read_text())
            for key,expected in [('joint_plan_hash',job['plan_hash']),('learning_setting',job['learning_setting']),('seed',job['seed']),('epochs',job['epochs']),('method',job.get('method','replay')),('joint_domain_mode',job.get('domain_mode','none'))]:
                if config.get(key)!=expected:raise ValueError('Run/manifest configuration mismatch: '+key)
            from .runner import job_config
            expected=job_config(job,plan,config['joint_plan_path'],
                config.get('client_training_concurrency',0),
                config.get('joint_local_evaluation',False),
                config.get('joint_mechanistic_probes',False),
                config.get('joint_checkpoint_policy','all'))
            if json.loads(json.dumps(expected.to_dict()))!=config:
                raise ValueError('Full execution configuration differs from declared manifest job')
            receipts=sorted((directory/'launch_receipts').glob('*.json'))
            if not receipts:raise ValueError('Missing mandatory launch receipts')
            for launch in receipts:
                record=json.loads(launch.read_text())
                if (record['config_signature']!=expected.signature() or record['plan_hash']!=job['plan_hash']
                        or record.get('job_id')!=job['job_id'] or not record.get('git_commit')
                        or record.get('tracked_worktree_dirty') is not False):
                    raise ValueError('Launch receipt configuration mismatch')
            identity_path=directory/'run_identity.json'
            if not identity_path.is_file():raise ValueError('Missing mandatory run identity')
            identity=json.loads(identity_path.read_text())
            revisions={json.loads(p.read_text()).get('git_commit') for p in receipts}
            if revisions!={identity['runtime'].get('git_revision')}:
                raise ValueError('Run identity and launch revisions differ')
            for key in ('source_sha256','dependencies','tracked_diff_sha256'):
                if not identity['runtime'].get(key):raise ValueError('Incomplete runtime identity: '+key)
            if not identity.get('dataset') or not identity.get('curriculum_sha256'):
                raise ValueError('Incomplete data or curriculum identity')
            history=json.loads((directory/'metrics.json').read_text())
            if len(history)!=len(plan['intervals']):raise ValueError('Incomplete metrics')
            if any(not r.get('sync_accuracy_report',{}).get('artifact_sha256') for r in history):
                raise ValueError('Missing mandatory per-sync artifact checksum receipts')
            assert_sync_reports_complete(directory,history)
            from .knowledge_audit import audit_knowledge
            ledger=audit_knowledge(directory,plan)
            row.update(acknowledged_occurrences=ledger['acknowledged'],pending_occurrences=ledger['pending'],
                       committed_versions=ledger['commits'])
            for r in plan['intervals']:
                interval_id=r['interval_id']
                if config.get('joint_mechanistic_probes'):
                    probe=json.loads((directory/f'joint_mechanistic_{interval_id:05d}.json').read_text())
                    if probe['interval']!=interval_id or probe['split']!='validation' or not probe['no_feedback']:
                        raise ValueError('Invalid mechanistic probe receipt')
                if config.get('joint_local_evaluation'):
                    communication=json.loads((directory/f'joint_communication_{interval_id:05d}.json').read_text())
                    if communication['interval']!=interval_id:raise ValueError('Invalid communication diagnostics receipt')
                record=json.loads((directory/f"joint_interval_{r['interval_id']:05d}.json").read_text())
                if record['server_state_digest_before']!=record['server_state_digest_after']:
                    raise ValueError('Server mutated inside interval')
                if [t for x in record['atomic_barriers'] for t in x.get('atomic_steps',[x['atomic_step']])]!=r['atomic_steps']:
                    raise ValueError('Executed atomic schedule mismatch')
                if {x['occurrence_id'] for x in record['local_occurrences']}!=set(r['occurrences']):
                    raise ValueError('Executed occurrence mismatch')
            if not (directory/'model_final.pt').exists():raise ValueError('Missing final model')
            summary=json.loads((directory/'summary.json').read_text())
            if summary.get('final_metrics')!=history[-1] or summary.get('job_id')!=job['job_id']:
                raise ValueError('Summary differs from authoritative terminal metrics')
            row['status']='artifact_verified'
            sync=directory/'per_sync_accuracy'/f'sync_{len(history):03d}.json'
            terminal=json.loads(sync.read_text())
            row.update(test_accuracy=terminal['test_accuracy'],
                validation_accuracy=terminal['validation_accuracy'],
                observed_classes=terminal['observed_class_count'],
                integrated_classes=terminal['server_integrated_class_count'],
                never_predicted=len(terminal['zero_prediction_classes']))
    except Exception as exc:
        row['status']='audit_failed';row['reason']=repr(exc)

    return row
