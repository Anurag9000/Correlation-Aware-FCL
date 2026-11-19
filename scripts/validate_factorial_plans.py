"""Validate both 32x32 backbones, complete 100-class math, and the two-job pilot."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.stream_audit import audit
from cafcl.full_matrix import expand


def main():
    root=Path(__file__).resolve().parents[1]
    frozen=json.loads((root/'configs/designs/stream_v13_first4_then2.json').read_text())
    frozen_probe=json.loads((root/'configs/designs/primary_probe_v13_first4_then2.json').read_text())
    snapshots={}
    for backbone in ('vit','cnn'):
        actual=manifest(model=backbone)
        assert actual['dataset']=='cifar100' and actual['batch_size']==1024
        assert actual['method_aggregator_pairs']==99
        probe=audited_plan(make_jobs(backbone=backbone))
        assert probe==frozen_probe[backbone], f'{backbone}: first4-then2 pilot drifted'
        assert probe['client_execution_mode']=='parallel_task_barriers'
        assert probe['client_models_per_run']==5
        assert probe['process_topology']=='two_independent_runs_on_one_gpu'
        assert probe['training_intervals']==9
        assert probe['initial_local_tasks']==2 and probe['subsequent_local_tasks']==1
        assert probe['total_jobs']==2
        assert [len(x['offline_classes']) for x in probe['offline_schedule']]==[4]+[2]*8
        assert actual==frozen['models'][backbone], f'{backbone}: changed v13 plan or checksum'
        assert actual['job_count']==32133
        snapshots[backbone]=actual
        print(f'{backbone}: {actual["job_count"]} feasible v13 jobs; sha256={actual["sha256"]}',flush=True)
    audit_result=audit(expand(include_invalid=True,model='vit'))
    assert audit_result==frozen['feasibility_audit'], 'v13 mathematical feasibility drift'
    assert audit_result['input_jobs']==53805
    assert audit_result['accepted_jobs']==32133
    assert audit_result['rejected_jobs']==21672
    print('Mathematical v13 audit: 53,805 candidates -> 32,133 feasible; 21,672 pruned',flush=True)
    print('Primary pilot: exactly two jobs; 9 train intervals 4-first/2-after; full global CL synchronization',flush=True)

if __name__=='__main__':main()
