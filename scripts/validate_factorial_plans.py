"""Validate both 32x32 backbones, complete 100-class math, and the two-job pilot."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.stream_audit import audit
from cafcl.full_matrix import expand


def main():
    root=Path(__file__).resolve().parents[1]
    frozen=json.loads((root/'configs/designs/stream_v14_calibrated.json').read_text())
    snapshots={}
    pilots={}
    for backbone in ('vit','cnn'):
        actual=manifest(model=backbone)
        assert actual['dataset']=='cifar100' and actual['batch_size']==64
        assert actual['method_aggregator_pairs']==99
        probe=audited_plan(make_jobs(backbone=backbone))
        assert probe['backbone']==actual['backbone']
        assert probe['client_execution_mode']=='parallel_task_barriers'
        assert probe['client_models_per_run']==5
        assert probe['process_topology']=='two_independent_runs_on_one_gpu'
        assert probe['training_intervals']==9
        assert probe['initial_local_tasks']==2 and probe['subsequent_local_tasks']==1
        assert probe['total_jobs']==2
        assert [len(x['offline_classes']) for x in probe['offline_schedule']]==[4]+[2]*8
        assert probe['micro_batch_size']==probe['batch_size']==64, 'Pilot must use a single physical batch, no accumulation'
        assert probe['sgd_momentum']==.9
        assert probe['sgd_weight_decay']==.0005
        assert probe['private_reservoir_per_class']==50
        assert actual==frozen['models'][backbone], 'v14 factorial registry changed'
        historical=frozen['pilots'][backbone]
        expected={**historical,'micro_batch_size':historical['batch_size'],
                  'job_ids':[x.replace('_V14_CALIBRATED_','_V15_NO_ACCUMULATION_') for x in historical['job_ids']]}
        assert probe==expected, 'v15 physical-batch pilot diverged from the otherwise frozen v14 plan'
        snapshots[backbone]=actual
        pilots[backbone]=probe
        print(f'{backbone}: {actual["job_count"]} feasible v14 jobs; sha256={actual["sha256"]}',flush=True)
    audit_result=audit(expand(include_invalid=True,model='vit'))
    assert audit_result==frozen['feasibility_audit'], 'v14 audit feasibility changed'
    assert audit_result['accepted_jobs']==snapshots['vit']['job_count']
    print('V14_DESIGN_SNAPSHOT='+json.dumps(dict(models=snapshots,pilots=pilots,
          feasibility_audit=audit_result),sort_keys=True),flush=True)
    print('Per-sync held-out audit: 10 reports for each job, CSV+MD+PNG+JSON',flush=True)
    print('Dual pilot: PHYSICAL BATCH 64, micro_batch_size=64, one full-batch forward/backward, no accumulation',flush=True)

if __name__=='__main__':main()
