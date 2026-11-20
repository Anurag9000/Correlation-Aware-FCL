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
from .upstream_fidelity import fidelity_report, require_verified

SUITES={
    'glfc':('native_glfc','native_glfc_cbam'),
    'lga': ('native_lga','native_lga_cbam'),
    'feat':('native_feat','native_feat_etf'),
}

def make_native_jobs(suite,*,seed=0,lr=.03,epochs=10,client_training_concurrency=0):
    if suite not in SUITES:raise ValueError('Unknown native suite')
    method,model=SUITES[suite]
    full,_=make_jobs()
    full=replace(full,seed=seed,lr=lr,epochs=epochs,training_augmentation='none',
                 method=method,model=model,
                 optimizer_state_policy='persist',
                 client_training_concurrency=client_training_concurrency,
                 job_id=f"{suite.upper()}_PILOT_SOURCE_CORE_{'V4' if suite=='feat' else 'V3'}_FULL_SEED{seed}")
    offline=replace(full,participation=.8,stream_offline_policy='rotate_once',
                    job_id=f"{suite.upper()}_PILOT_SOURCE_CORE_{'V4' if suite=='feat' else 'V3'}_ROTATING_OFFLINE_SEED{seed}")
    for cfg in (full,offline):cfg.validate()
    return full,offline

def main(argv=None):
    p=argparse.ArgumentParser(description='Native GLFC/LGA/FEAT concurrent full versus rotating-offline CIFAR-100 pilot')
    p.add_argument('--suite',choices=tuple(SUITES),required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--client-training-concurrency',type=int,default=0,
                   help='max active clients per simulation process; 0 means all five')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-dir',default='results/native_offline_pilots')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--permit-adapted-legacy',action='store_true',
                   help='Explicitly run the OLD incomplete native-loss adaptation, NOT the original algorithm')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--worker',type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.gpu!=0:p.error('This campaign is explicitly pinned to GPU 0')
    jobs=make_native_jobs(args.suite,seed=args.seed,lr=args.lr,epochs=args.epochs,
                          client_training_concurrency=args.client_training_concurrency)
    plan=audited_plan(jobs)
    plan.update(suite=args.suite,native_method=SUITES[args.suite][0],
                native_upstream_model=SUITES[args.suite][1],experiment='native_compatible_port',
                full_and_offline_same_gpu=0,concurrent_processes=2,
                source_fidelity='UNVERIFIED_PARTIAL_ADAPTATION',
                upstream_fidelity=fidelity_report(args.suite),
                original_native_train_transform=(
                    'official '+('GLFC/src/fl_main.py' if args.suite=='glfc' else
                    'LGA/fl_main.py')+':train_transform'
                    if args.suite in {'glfc','lga'} else None),
                original_feat_training_transform=(
                    'FEAT/methods/FEAT.py:_get_norm_and_transform(cifar100)'
                    if args.suite=='feat' else None))
    # The generic Replay planning helper reports Algorithm R/reservoir memory.
    # It is NOT executed by any of these three source-method pilot runners.
    # Replace each inherited assertion with the real method-specific policy.
    is_feat=args.suite=='feat'
    plan.update(
        replay_sampling=('authors_FLDataSelector_per_class_local_QR_SVD_with_replacement'
                         if is_feat else
                         'authors_entropy_triggered_previous_pair_herding'),
        replay_reservoir_fraction=None,
        inactive_generic_reservoir_fraction_config=jobs[0].replay_reservoir_fraction,
        private_reservoir_per_class=None,
        expected_server_exemplars_after_flush=None,
        source_memory_count_per_class=jobs[0].memory_per_class,
        memory_count_is_upper_bound=True,
        training_augmentation=('authors_FEAT_cifar_crop_flip_jitter'
                               if is_feat else
                               'authors_GLFC_LGA_cifar_crop_flip_jitter'),
        mathematical_source_training_optimizer_complete=False,
    )
    root=Path(args.output_dir)/args.suite/('seed'+str(args.seed))/( 'source_v4' if args.suite=='feat' else 'source_v3')
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'design.json',plan)
    if args.plan_only:
        print(json.dumps(plan,indent=2))
        return
    require_verified(args.suite, permit_legacy=args.permit_adapted_legacy)
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
          '--client-training-concurrency',str(args.client_training_concurrency),
          '--gpu','0','--data-root',args.data_root,'--output-dir',args.output_dir,
          '--no-download']
    if args.permit_adapted_legacy:base.append('--permit-adapted-legacy')
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
        f'Two matched {args.suite.upper()} source-method adapter/FedAvg runs,')
    report_text=report_text.replace(
        f"Training augmentation: {jobs[0].training_augmentation};",
        "Training augmentation: authors' original CIFAR crop/flip/jitter via PIL;")
    report_text += ('\\n**Method fidelity:** This is an adapted native-loss '
                    'pilot, not the full upstream published method. '
                    'See docs/NATIVE_OFFLINE_FIDELITY.md.\\n')
    report_path.write_text(report_text)
    print('PAIRED_RESULT',json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
