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
    for job in jobs:
        directory=root/job['job_id'];row={'job_id':job['job_id'],'status':'not_started'}
        try:
            status=directory/'status.json'
            if status.exists():row['status']=json.loads(status.read_text())['status']
            if row['status']=='completed':
                plan=json.loads((directory/'joint_plan.json').read_text())
                if plan['plan_hash']!=job['plan_hash'] or digest({k:v for k,v in plan.items() if k!='plan_hash'})!=job['plan_hash']:
                    raise ValueError('Run plan hash mismatch')
                history=json.loads((directory/'metrics.json').read_text())
                if len(history)!=len(plan['intervals']):raise ValueError('Incomplete metrics')
                assert_sync_reports_complete(directory,history)
                for r in plan['intervals']:
                    record=json.loads((directory/f"joint_interval_{r['interval_id']:05d}.json").read_text())
                    if record['server_state_digest_before']!=record['server_state_digest_after']:
                        raise ValueError('Server mutated inside interval')
                    if [x['atomic_step'] for x in record['atomic_barriers']]!=r['atomic_steps']:
                        raise ValueError('Executed atomic schedule mismatch')
                    if {x['occurrence_id'] for x in record['local_occurrences']}!=set(r['occurrences']):
                        raise ValueError('Executed occurrence mismatch')
                if not (directory/'model_final.pt').exists():raise ValueError('Missing final model')
                json.loads((directory/'summary.json').read_text())
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
        rows.append(row)
    report={'counts':dict(Counter(x['status'] for x in rows)),'jobs':rows,
            'scope':'Artifact completeness, not causal validity or independent checkpoint inference'}
    write_json(root/'joint_campaign_audit.json',report)
    keys=sorted({k for row in rows for k in row})
    with (root/'joint_campaign_audit.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return report
