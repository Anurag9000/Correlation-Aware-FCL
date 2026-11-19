"""Validate both 32x32 backbones, complete 100-class math, and the two-job pilot."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.stream_audit import audit
from cafcl.full_matrix import expand


def main():
    root=Path(__file__).resolve().parents[1]
    frozen=json.loads((root/'configs/designs/stream_v16_batch64.json').read_text())
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
        assert probe['batch_size']==64, 'Pilot must use one physical batch of 64'
        assert probe['sgd_momentum']==.9
        assert probe['sgd_weight_decay']==.0005
        assert probe['private_reservoir_per_class']==50
        assert actual==frozen['models'][backbone], 'v16 source-generated factorial registry changed'
        assert probe==frozen['pilots'][backbone], 'v16 paired pilot plan unexpectedly changed'
        snapshots[backbone]=actual
        pilots[backbone]=probe
        print(f'{backbone}: {actual["job_count"]} feasible v16 jobs; sha256={actual["sha256"]}',flush=True)
    audit_result=audit(expand(include_invalid=True,model='vit'))
    assert audit_result==frozen['feasibility_audit'], 'v16 feasibility audit changed'
    assert audit_result['accepted_jobs']==snapshots['vit']['job_count']
    print('V16_DESIGN_SNAPSHOT='+json.dumps(dict(models=snapshots,pilots=pilots,
          feasibility_audit=audit_result),sort_keys=True),flush=True)
    print('Per-sync held-out audit: 10 reports for each job, CSV+MD+PNG+JSON',flush=True)
    print('Dual pilot: single physical batch 64, Replay uses mixed current/historical data',flush=True)

if __name__=='__main__':main()
