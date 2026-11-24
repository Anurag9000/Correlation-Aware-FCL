"""Independent held-out inference from a trusted completed model checkpoint."""
from pathlib import Path
import argparse
import json
import math
import torch


def verify_final(run_dir,data_root='data',device='cpu',download=False):
    from .config import RunConfig
    from .data import get_data_bank
    from .identity import runtime_identity,dataset_identity
    from .models import create_model,task_class_sets_from_state_dict
    from .evaluation import evaluate
    from .joint.task_evaluation import evaluate_task_conditioned
    root=Path(run_dir)
    if json.loads((root/'status.json').read_text())['status']!='completed':
        raise ValueError('Only completed runs can be independently verified')
    cfg=RunConfig(**json.loads((root/'config.json').read_text()))
    cfg.validate()
    if not cfg.joint_plan_path and (cfg.scenario!='disjoint' or cfg.spurious_strength):
        raise ValueError('Verifier supports canonical joint and disjoint stream evaluation only')
    identity=json.loads((root/'run_identity.json').read_text())
    if identity['runtime']!=runtime_identity():
        raise RuntimeError('Verification requires the original source and dependency identity')
    bank=get_data_bank(cfg.dataset,data_root,download=download,seed=cfg.seed,class_subset=cfg.class_subset)
    if identity['dataset']!=dataset_identity(bank):
        raise RuntimeError('Verification dataset differs from the original raw data')
    # The caller must supply a trusted locally produced checkpoint.
    checkpoint=torch.load(root/'model_final.pt',map_location='cpu',weights_only=False)
    if json.dumps(checkpoint['configuration'],sort_keys=True)!=json.dumps(cfg.to_dict(),sort_keys=True):
        raise ValueError('Final checkpoint configuration mismatch')
    state=checkpoint['model_state']
    groups=task_class_sets_from_state_dict(state) if cfg.learning_setting=='task_il' else None
    model=create_model(cfg.model,bank.num_classes,allow_test_model=cfg.dataset=='synthetic',
                       task_class_sets=groups,learning_setting=cfg.learning_setting).to(device)
    model.load_state_dict(state)
    saved=json.loads((root/'metrics.json').read_text())[-1]
    observed=set(saved['observed_classes'])
    if groups is not None:groups={h:c for h,c in groups.items() if set(c)<=observed}
    checks={}
    def check(name,actual,expected):
        if expected is None:raise ValueError('Missing recorded metric: '+name)
        checks[name]={'recomputed':actual,'recorded':expected,
                      'matches':math.isclose(actual,expected,rel_tol=0.,abs_tol=1e-8)}
    for split in ('test','validation'):
        scores,_=evaluate(model,bank,observed,cfg.batch_size,device,
                           task_class_groups=groups,split=split,loader_seed=0)
        check(split+'_historical_accuracy',scores['accuracy'],saved['per_sync_'+split+'_accuracy'])
        if groups is not None:
            standard=evaluate_task_conditioned(model,bank,groups,cfg.batch_size,device,split)
            original=saved['standard_task_conditioned_'+split]
            check(split+'_standard_task_accuracy',standard['macro_head_accuracy'],original['macro_head_accuracy'])
            for head,record in standard['per_head'].items():
                check(split+':'+head,record['accuracy'],original['per_head'][head]['accuracy'])
    return {'verified':all(x['matches'] for x in checks.values()),
            'device':str(device),'checks':checks,
            'scope':'Independent inference, not reproduction of the training trajectory'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--device',default='cpu')
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    result=verify_final(a.run_dir,a.data_root,a.device,a.download)
    print(json.dumps(result,indent=2))
    if not result['verified']:raise SystemExit(1)


if __name__=='__main__':main()
