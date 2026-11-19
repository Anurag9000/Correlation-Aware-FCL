"""Generate a separate, batch-64 representative GPU pilot from registered jobs."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cafcl.planner import all_supported_suite


def pilot_features(cfg):
    values={
        'method':cfg.method,'aggregator':cfg.aggregator,'selector':cfg.selector,
        'scenario':cfg.scenario,'learning_setting':cfg.learning_setting,
        'participation_schedule':cfg.participation_schedule,
        'memory_policy':cfg.memory_policy,'pending_policy':cfg.pending_policy,
        'optimizer_state_policy':cfg.optimizer_state_policy,
        'boundary_mode':cfg.boundary_mode,
    }
    result={f'{key}={value}' for key,value in values.items()}
    if cfg.participation in {1.,.5,.25}:
        result.add(f'participation={cfg.participation:g}')
    flags={
        'full_sync':cfg.full_sync,
        'memory_reset':cfg.reset_memory_on_sync,
        'gradient_diagnostics':cfg.gradient_diagnostics,
        'causal_diagnostics':cfg.causal_diagnostics,
        'subset_diagnostics':cfg.subset_diagnostics,
        'prepost_diagnostics':cfg.prepost_diagnostics,
        'representation_diagnostics':cfg.representation_diagnostics,
        'spurious_ood_eval':cfg.spurious_ood_eval,
        'spurious_strength':cfg.spurious_strength>0,
        'label_noise':cfg.label_noise_rate>0,
        'heterogeneous_clients':cfg.client_heterogeneity,
        'checkpoint_resume':cfg.checkpoint_every>0,
        'sample_overlap':cfg.sample_overlap_fraction>0,
        'client_sample_skew':cfg.client_sample_skew>0,
        'class_imbalance':cfg.class_imbalance_power>0,
        'delayed_arrivals':bool(cfg.arrival_lag_by_client),
        'recurrence_period':cfg.recurrence_period>0,
        'participation_overrides':bool(cfg.participation_overrides),
        'class_allocation_override':bool(cfg.holders_override or cfg.class_holder_weights),
    }
    result.update(f'{key}=true' for key,value in flags.items() if value)
    return result


def required_features():
    result=set()
    for key,values in {
        'method':['sgd','gem','agem','replay','ewc','si','er_ace','der','derpp','mas','lwf'],
        'aggregator':['fedavg','uniform','fedavgm','fedadam','class_balanced',
                      'inverse_participation','conflict_aware','conflict_projection','retention_aware'],
        'selector':['random','fixed','round_robin','least_recent','coverage','old','new',
                    'balanced','diversity','overlap','related','recurrence','fairness',
                    'stochastic_related','historical_coverage','feature_max','feature_min',
                    'gradient_align','gradient_conflict','forgetting','gem_informed','history_adaptive'],
        'scenario':['disjoint','iid','non_iid','overlap','related','recurrence','asynchronous',
                    'mixed','domain','gradual_drift','shifted_recurrence'],
        'learning_setting':['class_il','task_il','domain_il','boundary_free'],
        'participation_schedule':['constant','increasing','decreasing','alternating','stage','feedback'],
        'memory_policy':['random_per_class','recent_per_class'],
        'pending_policy':['latest','retain_all','bounded'],
        'optimizer_state_policy':['reset','persist'],
        'boundary_mode':['known','hidden'],
        'participation':['1','0.5','0.25'],
    }.items():
        result.update(f'{key}={value}' for value in values)
    result.update({
        'full_sync=true','memory_reset=true','gradient_diagnostics=true',
        'causal_diagnostics=true','subset_diagnostics=true','prepost_diagnostics=true',
        'representation_diagnostics=true','spurious_ood_eval=true','spurious_strength=true',
        'label_noise=true','heterogeneous_clients=true','checkpoint_resume=true',
        'sample_overlap=true','client_sample_skew=true','class_imbalance=true',
        'delayed_arrivals=true','recurrence_period=true','participation_overrides=true',
        'class_allocation_override=true',
    })
    return result


def select_pilot_jobs(jobs):
    candidates=[(index,cfg,pilot_features(cfg)) for index,cfg in enumerate(jobs)
                if not cfg.fork_source_job and cfg.block_id!='L02']
    # No registered production arm currently checkpoints every round. The pilot
    # opts one representative job into checkpointing after selecting it; the
    # interrupted-resume behavior is separately exercised by test_advanced.py.
    remaining=required_features()-{'checkpoint_resume=true'}
    selected=set()
    while remaining:
        ranked=[]
        for index,cfg,features in candidates:
            if index in selected:continue
            gain=len(features & remaining)
            if gain:
                cost=cfg.stages*cfg.num_clients+12*sum((cfg.gradient_diagnostics,
                    cfg.causal_diagnostics,cfg.subset_diagnostics,cfg.representation_diagnostics))
                ranked.append((gain/cost,gain,-cost,-index,index,features))
        if not ranked:
            raise RuntimeError(f'No registered CIFAR-10 job covers: {sorted(remaining)}')
        _,_,_,_,index,features=max(ranked)
        selected.add(index)
        remaining-=features
    # Keep a complete matched baseline/include/exclude trajectory fork.
    fork=[(index,cfg) for index,cfg in enumerate(jobs)
          if cfg.job_id in {'L02_forkbaseline_related_sgd',
                            'L02_forkinclude_related_sgd',
                            'L02_forkexclude_related_sgd'}]
    if len(fork)!=3:
        raise RuntimeError('Expected one complete related-SGD counterfactual fork cohort')
    selected.update(index for index,_ in fork)
    return [(index,cfg) for index,cfg in enumerate(jobs) if index in selected]


def pilot_config(cfg):
    pilot_id='PILOT_'+cfg.job_id
    changes={
        'job_id':pilot_id,'rounds_per_stage':2,'epochs':1,
        'train_fraction':.15,'require_cuda':True,
    }
    if cfg.fork_source_job:
        changes.update(fork_source_job='PILOT_'+cfg.fork_source_job,
                       fork_checkpoint_round=2,intervention_round=2)
    elif cfg.force_target!='none' and cfg.intervention_round>=0:
        changes['intervention_round']=min(cfg.intervention_round,
                                          cfg.stages*2-1)
    result=replace(cfg,**changes)
    result.validate()
    if result.batch_size!=64:
        raise AssertionError(f'Pilot batch size changed: {result.job_id}')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/cifar10_gpu_pilot_v1/job_plan.json')
    args=parser.parse_args()
    registered=all_supported_suite('cifar10',0)
    selected=select_pilot_jobs(registered)
    jobs=[pilot_config(cfg) for _,cfg in selected]
    checkpoint_index=next((i for i,cfg in enumerate(jobs) if cfg.method=='sgd' and
                           cfg.aggregator=='fedavg' and cfg.scenario in {'iid','disjoint'}),None)
    if checkpoint_index is None:
        checkpoint_index=next(i for i,cfg in enumerate(jobs) if cfg.method=='sgd')
    jobs[checkpoint_index]=replace(jobs[checkpoint_index],checkpoint_every=1)
    jobs[checkpoint_index].validate()
    ids=[cfg.job_id for cfg in jobs]
    if len(ids)!=len(set(ids)):
        raise RuntimeError('Pilot job identifiers are not unique')
    coverage=set().union(*(pilot_features(cfg) for cfg in jobs))
    missing=required_features()-coverage
    if missing:
        raise RuntimeError(f'Pilot coverage is incomplete: {sorted(missing)}')
    position={cfg.job_id:index for index,cfg in enumerate(jobs)}
    late=[(cfg.job_id,cfg.fork_source_job) for cfg in jobs
          if cfg.fork_source_job and position[cfg.fork_source_job]>=position[cfg.job_id]]
    if late:
        raise RuntimeError(f'Pilot fork source ordering is invalid: {late}')
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'dataset':'cifar10','suite':'representative_gpu_pilot',
             'batch_size':64,'training_fraction_of_eligible_data':.15,
             'rounds_per_stage':2,'registered_source_job_count':len(registered),
             'covered_features':sorted(coverage),
             'coverage_warning':'Pilot results are not production results.',
             'jobs':[cfg.to_dict() for cfg in jobs]}
    path.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Prepared {len(jobs)} independent CUDA pilot jobs covering '
          f'{len(coverage)} registered mechanisms; batch size 64; output={path}')


if __name__=='__main__':
    main()
