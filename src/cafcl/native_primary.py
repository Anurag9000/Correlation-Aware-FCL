"""Three independent GLFC/LGA/FEAT paired CIFAR-100 pilots.

Each suite launches BOTH arms as independent concurrent processes on GPU 0.
All five clients train in both arms. Only communication eligibility differs.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
# The physical CUDA device selection is fixed before any torch/CUDA import.
os.environ['CUDA_VISIBLE_DEVICES']='0'
from dataclasses import replace
from pathlib import Path
from .primary_probe import make_jobs,audited_plan,build_report
from .simulator import run_experiment,atomic_json

SUITES={
    'glfc':('native_glfc','native_glfc_cbam'),
    'lga': ('native_lga','native_lga_cbam'),
    'feat':('native_feat','native_feat_resnet'),
}

def make_native_jobs(suite,*,seed=0,lr=.03,epochs=10):
    if suite not in SUITES:raise ValueError('Unknown native suite')
    method,model=SUITES[suite]
    full,_=make_jobs(backbone='cnn',seed=seed,lr=lr,epochs=epochs,replay_fraction=1/9)
    full=replace(full,method=method,model=model,
                 job_id=f'{suite.upper()}_PILOT_FULL_V1_SEED{seed}')
    offline=replace(full,participation=.8,stream_offline_policy='rotate_once',
                    job_id=f'{suite.upper()}_PILOT_ROTATING_OFFLINE_V1_SEED{seed}')
    for cfg in (full,offline):cfg.validate()
    return full,offline

def main(argv=None):
    p=argparse.ArgumentParser(description='Native GLFC/LGA/FEAT concurrent full versus rotating-offline CIFAR-100 pilot')
    p.add_argument('--suite',choices=tuple(SUITES),required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-dir',default='results/native_offline_pilots')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--worker',type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.gpu!=0:p.error('This campaign is explicitly pinned to GPU 0')
    jobs=make_native_jobs(args.suite,seed=args.seed,lr=args.lr,epochs=args.epochs)
    plan=audited_plan(jobs)
    plan.update(suite=args.suite,native_method=SUITES[args.suite][0],
                native_upstream_model=SUITES[args.suite][1],experiment='native_compatible_port',
                full_and_offline_same_gpu=0,concurrent_processes=2,
                source_fidelity='native_losses_and_backbones; see FIDELITY.md for deviations')
    root=Path(args.output_dir)/args.suite/('seed'+str(args.seed))
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'design.json',plan)
    if args.plan_only:
        print(json.dumps(plan,indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; production pilots cannot silently run on CPU')
    if args.worker is not None:
        result=run_experiment(jobs[args.worker],str(root/'runs'),args.data_root,download=False)
        print(json.dumps(result,indent=2),flush=True)
        return
    # Download once; read-only immutable dataset shared by the two workers.
    from .data import get_data_bank
    get_data_bank('cifar100',args.data_root,download=not args.no_download,seed=args.seed)
    base=[sys.executable,'-m','cafcl.native_primary','--suite',args.suite,
          '--seed',str(args.seed),'--lr',str(args.lr),'--epochs',str(args.epochs),
          '--gpu','0','--data-root',args.data_root,'--output-dir',args.output_dir,
          '--no-download']
    handles=[]
    codes=[]
    try:
        for idx,cfg in enumerate(jobs):
            env=os.environ.copy()
            env['CUDA_VISIBLE_DEVICES']='0'
            env.setdefault('OMP_NUM_THREADS','1')
            env.setdefault('MKL_NUM_THREADS','1')
            target=root/(f'job_{idx}.log')
            log=target.open('a')
            process=subprocess.Popen(base+['--worker',str(idx)],env=env,
                                     stdout=log,stderr=subprocess.STDOUT)
            handles.append((process,log))
            print(f'Launched {cfg.job_id} PID={process.pid} GPU=0 log={target}',flush=True)
        while any(process.poll() is None for process,_ in handles):
            failed=[process for process,_ in handles if process.poll() not in (None,0)]
            if failed:
                for sibling,_ in handles:
                    if sibling.poll() is None:sibling.terminate()
                for sibling,_ in handles:
                    if sibling.poll() is None:
                        try:sibling.wait(timeout=10)
                        except subprocess.TimeoutExpired:sibling.kill();sibling.wait()
                break
            time.sleep(.25)
        codes=[process.wait() for process,_ in handles]
    finally:
        for process,log in handles:
            if process.poll() is None:process.terminate()
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()
    if any(code!=0 for code in codes):
        raise RuntimeError(f'Native {args.suite} paired workers failed: {codes}. Inspect {root}/job_*.log')
    comparison=build_report(root,jobs)
    if comparison is None:
        raise RuntimeError('Both workers finished but paired result files were not produced')
    # The shared paired reporter originated as the Replay pilot. Correct its
    # human-facing label for these distinct protocol-harmonized native methods.
    report_path=root/'comparison'/'comparison.md'
    report_text=report_path.read_text()
    report_text=report_text.replace('Two matched Replay/FedAvg runs,',
        f'Two matched {args.suite.upper()} native-loss/FedAvg adapted runs,')
    report_text += ('\\n**Method fidelity:** This is an adapted native-loss '
                    'pilot, not the full upstream published method. '
                    'See docs/NATIVE_OFFLINE_FIDELITY.md.\\n')
    report_path.write_text(report_text)
    print('PAIRED_RESULT',json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
