"""Enumerate and report the exact feasible/removed CIFAR-100 jobs."""
import argparse,json
from cafcl.full_matrix import expand
from cafcl.stream_audit import audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=None)
    p.add_argument('--backbone',choices=['vit','cnn'],default='vit')
    args=p.parse_args()
    # include_invalid allows reporting discarded configurations explicitly.
    report=audit(expand(seeds=(args.seed,) if args.seed is not None else (0,1,2),
                        model=args.backbone,include_invalid=True))
    print(json.dumps(report,indent=2))
