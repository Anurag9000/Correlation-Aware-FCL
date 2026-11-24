"""Audit-gated cross-experiment comparison, with explicit estimand boundaries.

Pairs are declared from the existing catalog, not selected by test results.
Multiple independent manifests may supply seed replications. Missing arms and
null diagnostics remain missing; aliases do not increase the sample size.
"""
from collections import defaultdict, Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .analysis import analyze_run
from .curriculum import digest
from .result_audit import audit_job
from .runner import write_json


METRICS=('terminal_test_accuracy','terminal_integrated_class_peak_drop',
         'terminal_integrated_task_head_peak_drop','mean_current_task_test_accuracy',
         'mean_historical_noncurrent_class_test_accuracy','mean_local_acquisition',
         'mean_signed_local_retention_drop','mean_integration_latency_intervals',
         'terminal_prediction_coverage','optimizer_steps','model_tensor_bytes')


def default_contrasts():
    rows=[]
    def add(name,a,b,match):
        rows.append({'contrast_id':name,'reference':a,'treatment':b,'expected_match':match})
    for s in ('S1','S2','S3'):
        for o in (0,1):
            key=f'{s}-O{o}'
            for depth in (2,3,5,10,20):
                a=f'H-{key}-L1'
                b=f'H-{key}-L{depth}' if depth<=5 else f'EXT-H-{key}-L{depth}'
                add(f'delayed-{key}-L{depth}',a,b,'atomic_data_and_epochs')
            for depth in (2,5):
                ref=f'H-{key}-L{depth}'
                add(f'no-replay-{key}-L{depth}',ref,f'CTRL-NOREPLAY-{key}-L{depth}','atomic_data_and_epochs')
                add(f'pooled-{key}-L{depth}',ref,f'CTRL-POOLED-{key}-L{depth}','atomic_data_and_epochs')
            add(f'centralized-{key}',f'H-{key}-L5',f'CTRL-CENTRAL-{key}','fresh_assignment_multiset_and_epochs')
            # Local-only is excluded from global endpoint contrasts; it has
            # no learned server. It is exported in the condition table instead.
            for n in (2,5,10):
                add(f'intratask-{key}-S{n}',f'INTRA-{key}-S1',f'INTRA-{key}-S{n}','atomic_data_and_epochs')
            if s in ('S1','S3') and o==0:
                for mode in ('client_fixed','client_periodic','client_corruption'):
                    add(f'domain-{key}-{mode}',f'H-{key}-L2',f'EXT-P-{s}-{mode}','atomic_data_and_epochs')
    add('semantic-relatedness','EXT-P-SEMANTIC-unrelated','EXT-P-SEMANTIC-related','class_occurrence_counts')
    return rows


def assignment_multiset_hash(plan):
    # Ignore placement only for the explicit centralized control. Preserve
    # each occurrence's class set and exact raw-index membership, not just
    # total unique class/sample counts. Undo epoch fragmentation first.
    slicing=plan['allocation_audit'].get('epoch_slicing',{})
    fragments=slicing.get('fragment_metadata',{})
    rows=[]
    for task in plan['tasks']:
        if fragments and fragments[task['occurrence_id']]['segment_index']!=0:continue
        rows.append(digest([task['classes'],sorted(task['train_indices'])]))
    return digest(sorted(rows))


def planned_budget(plan,epochs):
    audit=plan['allocation_audit'];slicing=audit.get('epoch_slicing',{})
    return {'parent_atomic_hash':slicing.get('parent_atomic_hash',plan['atomic_hash']),
            'assignment_multiset_hash':assignment_multiset_hash(plan),
            'training_target_hash':audit['training_target_hash'],
            'class_occurrence_counts':plan['overlap_audit']['class_occurrences'],
            'parent_epochs':epochs*slicing.get('segments',1),
            'planned_fresh_draws':audit['assignments']*epochs*slicing.get('segments',1),
            'communication_opportunities':sum(bool(x['upload_clients']) for x in plan['intervals']),
            'planned_client_downloads':sum(len(x['upload_clients']) for x in plan['intervals']),
            'unique_training_samples':audit['unique_assigned']}


def match_status(a,b,contract):
    if a['training_target_hash']!=b['training_target_hash']:return False
    if contract=='atomic_data_and_epochs':
        return a['parent_atomic_hash']==b['parent_atomic_hash'] and a['parent_epochs']==b['parent_epochs']
    if contract=='fresh_assignment_multiset_and_epochs':
        return a['assignment_multiset_hash']==b['assignment_multiset_hash'] and a['parent_epochs']==b['parent_epochs']
    if contract=='class_occurrence_counts':
        return a['class_occurrence_counts']==b['class_occurrence_counts'] and a['parent_epochs']==b['parent_epochs']
    raise ValueError('Unknown comparison match contract: '+contract)


def aggregate(values):
    n=len(values)
    if not n:return {'n':0,'mean':None,'sd':None,'ci95_low':None,'ci95_high':None}
    center=mean(values)
    if n==1:return {'n':1,'mean':center,'sd':None,'ci95_low':None,'ci95_high':None}
    from scipy.stats import t
    sd=stdev(values);half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
    return {'n':n,'mean':center,'sd':sd,'ci95_low':center-half,'ci95_high':center+half}


def read_condition(job,root,plan):
    audited=audit_job(job,root)
    row={'job_id':job['job_id'],'seed':job['seed'],'setting':job['learning_setting'],
         'family':job.get('family'),'status':audited['status']}
    if audited['status']!='artifact_verified':
        row['reason']=audited.get('reason')
        return row,None
    try:
        summary=analyze_run(root/job['job_id'],figures=False)
        term=summary['terminal'];resource=summary['resource_totals']
        row.update(control_kind=summary['control_kind'],
            terminal_test_accuracy=term['test_accuracy'],
            terminal_integrated_class_peak_drop=term['integrated_class_peak_drop'],
            terminal_integrated_task_head_peak_drop=term['integrated_task_head_peak_drop'],
            terminal_prediction_coverage=term['coverage'],
            terminal_private_macro_client_accuracy=summary['terminal_private_macro_client_accuracy'],
            optimizer_steps=resource['optimizer_steps'],
            model_tensor_bytes=(resource['model_upload_tensor_bytes']+resource['model_download_tensor_bytes']
                if resource['model_upload_tensor_bytes'] is not None and resource['model_download_tensor_bytes'] is not None else None))
        for key in ('mean_current_task_test_accuracy','mean_historical_noncurrent_class_test_accuracy',
                    'mean_local_acquisition','mean_signed_local_retention_drop','mean_integration_latency_intervals'):
            row[key]=summary[key]
        if summary['control_kind']=='local_only':
            row['untrained_server_test_accuracy']=row.pop('terminal_test_accuracy')
            # Do not compare an untrained server against trained FL models.
            for key in METRICS:
                if key.startswith(('terminal_','mean_current_task','mean_historical')):row[key]=None
        budget=planned_budget(plan,job['epochs'])
        row.update({k:v for k,v in budget.items() if k!='class_occurrence_counts'})
        return row,budget
    except Exception as exc:
        row['status']='analysis_failed';row['reason']=repr(exc)
        return row,None


def csv_rows(path,rows):
    keys=sorted({k for row in rows for k in row})
    with path.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def compare_campaigns(campaigns,output,contrasts_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    contrasts=json.loads(Path(contrasts_path).read_text()) if contrasts_path else default_contrasts()
    if len({x['contrast_id'] for x in contrasts})!=len(contrasts):raise ValueError('Duplicate contrast IDs')
    conditions=[];lookup={};known=set()
    for manifest,root in campaigns:
        manifest=Path(manifest).resolve();root=Path(root).resolve()
        catalog=json.loads(manifest.read_text())
        aliases=defaultdict(list)
        for alias in catalog.get('aliases',[]):aliases[alias['canonical_job_id']].append(alias['job_id'])
        for job in catalog['jobs']:
            ids=[job['job_id']]+aliases[job['job_id']]
            keys=[(job['seed'],name) for name in ids]
            if known.intersection(keys):
                raise ValueError('Duplicate seed/condition across supplied campaigns; choose one authoritative run')
            known.update(keys)
            plan=json.loads((manifest.parent/job['plan_file']).read_text())
            row,budget=read_condition(job,root,plan)
            row['result_root']=str(root)
            conditions.append(row)
            for key in keys:lookup[key]=(row,budget)
    pairs=[];missing=[];groups=defaultdict(list)
    for seed in sorted({seed for seed,_ in known}):
        for setting in ('task_il','class_il'):
            for c in contrasts:
                ids=[f"J-{c[k]}-{setting}" for k in ('reference','treatment')]
                ref,arm=[lookup.get((seed,id)) for id in ids]
                identity={'contrast_id':c['contrast_id'],'seed':seed,'setting':setting,
                          'reference':ids[0],'treatment':ids[1]}
                if ref is None or arm is None or ref[1] is None or arm[1] is None:
                    missing.append({**identity,'reason':'missing_or_unverified_arm'});continue
                a,ab=ref;b,bb=arm
                if a['job_id']==b['job_id'] and a['result_root']==b['result_root']:
                    missing.append({**identity,'reason':'same_canonical_condition'});continue
                matched=match_status(ab,bb,c['expected_match'])
                pair={**identity,'expected_match':c['expected_match'],'declared_match_verified':matched,
                      'fresh_draws_equal':ab['planned_fresh_draws']==bb['planned_fresh_draws'],
                      'communication_opportunities_equal':ab['communication_opportunities']==bb['communication_opportunities'],
                      'client_download_count_equal':ab['planned_client_downloads']==bb['planned_client_downloads'],
                      'optimizer_steps_equal':None if a['optimizer_steps'] is None or b['optimizer_steps'] is None else a['optimizer_steps']==b['optimizer_steps']}
                for metric in METRICS:
                    x=a.get(metric);y=b.get(metric)
                    delta=None if x is None or y is None else y-x
                    pair['delta_'+metric]=delta
                    if matched and delta is not None:groups[(c['contrast_id'],setting,metric)].append(delta)
                pairs.append(pair)
    estimates=[{'contrast_id':name,'setting':setting,'metric':metric,**aggregate(values)}
               for (name,setting,metric),values in sorted(groups.items())]
    csv_rows(out/'conditions.csv',conditions);csv_rows(out/'paired_differences.csv',pairs)
    csv_rows(out/'paired_estimates.csv',estimates);csv_rows(out/'missing_pairs.csv',missing)
    report={'condition_status_counts':dict(Counter(x['status'] for x in conditions)),
            'pairs':len(pairs),'verified_declared_matches':sum(x['declared_match_verified'] for x in pairs),
            'missing_or_aliased_pairs':len(missing),'estimates':len(estimates),
            'scope':'Descriptive paired differences. Student-t intervals require independent seed pairs; no inference for n=1, no multiplicity adjustment. Matching flags do not establish causal identification.'}
    write_json(out/'comparison_summary.json',report)
    return report
