"""Generate explicit matched remediation contrasts from an existing joint job.

No training or test evaluation is performed. Plan provenance and intervention
fields are retained. A match of optimizer steps and fresh draws is not a
claim of equal total FLOPs when historical replay draws differ.
"""
from pathlib import Path
from itertools import product
import argparse
import json
from .curriculum import compile_plan
from .adapter import load_plan
from .runner import write_json,job_config


def build_controls(job,plan,targets,output,*,fixed_support=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    setting=job['learning_setting']
    if job.get('method','replay')!='replay' or plan['allocation_audit'].get('control_kind','federated')!='federated':
        raise ValueError('Select an ordinary federated Replay base job')
    audit=plan['allocation_audit']
    if audit.get('epoch_slicing'):
        raise ValueError('Use a parent atomic plan, not an epoch-fragmented plan')
    if all(len(r['upload_clients'])==plan['num_clients'] for r in plan['intervals']):
        raise ValueError('Select a partial-participation base plan for archive contrasts')
    jobs=[];contrasts=[];names=set()
    standard={**job,'family':'REMEDIATION','families':['REMEDIATION'],
        'method':'replay','share_continual_state':True,'server_memory_per_class':50,
        'joint_archive_upload_policy':'selected','joint_archive_download_policy':'selected',
        'stale_update_policy':'damped_delta','terminal_stale_guard':False,
        'optimizer_state_policy':'persist','aggregation_weight_policy':'fresh_examples',
        'task_head_weight_policy':'historical_exposure',
        'physical_batch_policy':'fixed_pad','batch_size':64,
        'replay_ratio':1.,'replay_update_policy':'sampled_mixed_accumulated',
        'replay_class_sampling_policy':'class_uniform_with_replacement'}
    def add(name,p,changes=None):
        if name in names:raise ValueError('Duplicate control name')
        names.add(name)
        relative='plans/'+p['plan_hash']+'.json';write_json(out/relative,p)
        condition={**standard,**(changes or {}),'job_id':'J-'+name+'-'+setting,
            'plan_file':relative,'plan_hash':p['plan_hash'],'atomic_hash':p['atomic_hash'],
            'communication_intervals':len(p['intervals']),'task_occurrences':len(p['tasks']),
            'metadata':{'parent_job_id':job['job_id'],'control':name},'status':'planned'}
        job_config(condition,p,out/relative).validate()
        jobs.append(condition)
        return name
    def pair(ref,arm,contract='atomic_data_and_epochs',**extra):
        contrasts.append({'contrast_id':ref+'-vs-'+arm,'reference':ref,'treatment':arm,
                          'expected_match':contract,**extra})
    baseline=add('CORR-base',plan)
    for upload,download in product(('selected','all'),('selected','all')):
        if (upload,download)==('selected','selected'):continue
        name=add('CORR-archive-'+upload+'-'+download,plan,
            {'joint_archive_upload_policy':upload,'joint_archive_download_policy':download})
        pair(baseline,name)
    variants={
        'private':{'share_continual_state':False,'server_memory_per_class':0},
        'archive-no-download':{'joint_archive_download_policy':'none'},
        'append-only':{'server_memory_per_class':0},
        'undamped':{'stale_update_policy':'undamped_delta'},
        'direct':{'stale_update_policy':'direct_model'},
        'terminal-guard':{'terminal_stale_guard':True},
        'reset-sync':{'optimizer_state_policy':'reset_on_sync'},
        'reset-task':{'optimizer_state_policy':'reset_on_task'},
        'step-weight':{'aggregation_weight_policy':'optimizer_steps'},
        'variable-batch':{'physical_batch_policy':'variable'},
        'separate-replay-steps':{'replay_update_policy':'sampled_mixed'},
    }
    if setting=='task_il':
        variants.update({'head-measured':{'task_head_weight_policy':'measured_exposure'},
                         'head-equal':{'task_head_weight_policy':'equal_contributors'}})
    for name,changes in variants.items():pair(baseline,add('CORR-'+name,plan,changes))
    no_replay=add('CORR-no-replay',plan,{'method':'sgd','share_continual_state':False,
        'server_memory_per_class':0,'replay_ratio':0.})
    pair(baseline,no_replay,require_matched_compute=True,
         match_scope='optimizer steps and fresh draws; replay exposure and total forwards differ by design')
    if fixed_support is not None:
        if audit['raw_sample_overlap'] or audit.get('allocation_weights'):
            raise ValueError('Fixed support requires a disjoint unweighted allocation')
        p=compile_plan([(x['atomic_step'],x['client_id'],x['classes']) for x in plan['tasks']],
            targets,num_clients=plan['num_clients'],num_classes=plan['num_classes'],
            seed=plan['seed'],head_policy=plan['head_policy'],
            require_all_classes=not job.get('allow_partial_coverage',False),
            interval_boundaries=[max(r['atomic_steps'])+1 for r in plan['intervals'] if not r['terminal_flush']],
            upload_cohorts=[r['upload_clients'] for r in plan['intervals'] if not r['terminal_flush']],
            terminal_flush=plan['intervals'][-1]['terminal_flush'],
            allocation_order=audit.get('allocation_order','timeline'),
            minimum_samples=audit.get('allocation_minimum_samples',1),
            samples_per_class_occurrence=fixed_support).to_dict()
        pair(baseline,add('CORR-fixed-support',p),'class_occurrence_counts',
             match_scope='same occurrences, not same unique-data budget')
    write_json(out/'manifest.json',{'dataset':'cifar100','suite':'audit_remediation',
        'jobs':jobs,'aliases':[],'infeasible':[],'source_job_id':job['job_id'],
        'warning':'Planned controls, no completed training. Reserved confirmation slots remain unspecified.'})
    write_json(out/'contrasts.json',contrasts)
    return jobs,contrasts


def main():
    from ..data import get_data_bank
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--job-id',required=True)
    p.add_argument('--output',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--fixed-support',type=int);p.add_argument('--download',action='store_true')
    a=p.parse_args();path=Path(a.manifest)
    jobs=json.loads(path.read_text())['jobs']
    matches=[x for x in jobs if x['job_id']==a.job_id]
    if len(matches)!=1:raise ValueError('Expected exactly one base job')
    job=matches[0];bank=get_data_bank('cifar100',a.data_root,download=a.download,seed=job['seed'])
    plan=load_plan(path.parent/job['plan_file'],job['plan_hash'],bank.train.targets)
    result=build_controls(job,plan,bank.train.targets,a.output,fixed_support=a.fixed_support)
    print(json.dumps({'jobs':len(result[0]),'contrasts':len(result[1]),'training_launched':False}))


if __name__=='__main__':main()
