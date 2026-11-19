"""Single-run command and CIFAR-100-only task-stream sweep entrypoint."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .config import RunConfig
from .simulator import run_experiment


def main():
    if len(sys.argv)>1 and sys.argv[1]=='suite':
        from .full_matrix import main as stream_main
        return stream_main(sys.argv[2:])
    p=argparse.ArgumentParser(prog='cafcl')
    sub=p.add_subparsers(dest='action',required=True)
    single=sub.add_parser('run',help='Run one validated CIFAR-100 task-stream JSON configuration')
    single.add_argument('config');single.add_argument('--output-dir',default='results')
    single.add_argument('--data-root',default='data');single.add_argument('--download',action='store_true')
    opts=p.parse_args()
    config=RunConfig(**json.loads(Path(opts.config).read_text()))
    if config.dataset!='cifar100' or not config.stream_protocol:
        p.error('Retired datasets and stage-based job configurations are not launchable')
    print(json.dumps(run_experiment(config,opts.output_dir,opts.data_root,opts.download),indent=2))


if __name__=='__main__':main()
