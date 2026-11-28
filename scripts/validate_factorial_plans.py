"""Validate the factorial registry and fixed primary Replay/FedAvg design."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest,expand
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.stream_audit import audit


def main():
    root=Path(__file__).resolve().parents[1]
    frozen=json.loads((root/'configs/designs/stream_v16_batch64.json').read_text())
    snapshots={}
    for backbone in ('vit','cnn'):
        actual=manifest(model=backbone)
        assert actual['dataset']=='cifar100' and actual['batch_size']==64
        assert actual['method_aggregator_pairs']==99
        assert actual==frozen['models'][backbone], 'v16 factorial registry changed'
        snapshots[backbone]=actual
        print(f'{backbone}: {actual["job_count"]} feasible v16 jobs; sha256={actual["sha256"]}',flush=True)
    probe=audited_plan(make_jobs())
    assert probe['backbone']=='resnet18_cifar32'
    assert probe['optimizer_state_policy']=='persist'
    assert probe['training_augmentation']=='cifar_crop_flip'
    assert probe['client_training_concurrency']==5
    assert probe['training_intervals']==20 and probe['total_jobs']==2
    assert probe['private_reservoir_per_class']==50
    audit_result=audit(expand(include_invalid=True,model='vit'))
    assert audit_result==frozen['feasibility_audit'], 'v16 feasibility audit changed'
    assert audit_result['accepted_jobs']==snapshots['vit']['job_count']
    print('DESIGN_SNAPSHOT='+json.dumps(dict(models=snapshots,primary_probe=probe,
          feasibility_audit=audit_result),sort_keys=True),flush=True)
    print('Primary FedAvg: fixed crop/flip ResNet-18, batch 64, Replay, persistent SGD momentum, seed 0.',flush=True)

if __name__=='__main__':main()
