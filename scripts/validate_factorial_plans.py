"""Validate both 32x32 backbones, complete 100-class math, and the two-job pilot."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.stream_audit import audit
from cafcl.full_matrix import expand


def main():
    root=Path(__file__).resolve().parents[1]
    frozen=json.loads((root/'configs/designs/stream_v6_native32.json').read_text())
    for backbone in ('vit','cnn'):
        actual=manifest(model=backbone)
        expected=frozen['models'][backbone]
        assert actual==expected, f'{backbone}: planned jobs or hash drifted; regenerate only after audit'
        assert actual['dataset']=='cifar100' and actual['batch_size']==1024
        assert actual['method_aggregator_pairs']==99
        assert actual['job_count']==38370
        probe=audited_plan(make_jobs(backbone=backbone))
        assert probe==frozen['primary_probe_'+backbone]
        assert probe['total_jobs']==2 and probe['training_intervals']==5
        assert all(len(x['offline_classes'])==4 for x in probe['offline_schedule'])
        print(f'{backbone}: {actual["job_count"]} feasible jobs, batch 1024, 99/99 method pairs, '
              f'100 unique classes, sha256={actual["sha256"]}')
    audit_result=audit(expand(include_invalid=True,model='vit'))
    assert audit_result==frozen['feasibility_audit']
    assert audit_result['input_jobs']==53805
    assert audit_result['rejected_jobs']==15435
    assert audit_result['accepted_jobs']==38370
    print('Mathematical audit: 53,805 old jobs -> 38,370 feasible; 15,435 explicitly pruned')
    print('Primary pilot: exactly two jobs, single seed, one complete 100-class curriculum')

if __name__=='__main__':main()
