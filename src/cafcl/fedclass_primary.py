"""FedCLASS cross-entropy/FedAvg pilot: fully connected versus rotating offline.

Uses the exact vendored FedCLASS ResNet8 and SimpleLinear, plus the audited
CIFAR-100 v16 class stream and federation engine. Two independent experiments
launch CONCURRENTLY on physical CUDA device 0 (one process per arm).
This is a deliberately documented class-incremental adaptation, not the
upstream FedCLASS script's different task allocation and client scheduling.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

# Hard pin before importing any module that imports torch.
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from .primary_probe import make_jobs, audited_plan, build_report
from .simulator import atomic_json, run_experiment
from .upstream_fidelity import fidelity_report, require_verified

SOURCE_COMMIT="e03e30653165b76ddf980393794f395f838ee967"
MEMORY_MODES=("authors_exemplars","none")


def make_fedclass_jobs(*,seed=0,lr=.03,epochs=10,memory_mode="authors_exemplars",
                       replay_ratio=1.,client_training_concurrency=0):
    """Build matched FedCLASS CE/FedAvg class-IL jobs.

    authors_exemplars: execute original FedCLASS BaseLearner first-k selection;
    none: original CE/FedAvg without memory or historical replay.
    In either case, only communication visibility differs across the pair.
    """
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"Unknown memory mode: {memory_mode}")
    if not (0. <= replay_ratio <= 10.):
        raise ValueError("Replay ratio must be finite and between 0 and 10")
    if memory_mode=="authors_exemplars" and replay_ratio!=1.:
        raise ValueError("The original source CE does not have a tunable replay ratio; only compatibility value 1 is accepted")
    if memory_mode=="none" and replay_ratio!=1.:
        raise ValueError("No-replay mode must use the default replay ratio")
    has_replay=memory_mode=="authors_exemplars"
    fraction=0. # No Algorithm R in the FedCLASS source; author picks first k.
    baseline,_=make_jobs()
    baseline=replace(baseline,seed=seed,lr=lr,epochs=epochs,
                     training_augmentation='none')
    tag="AUTHORS_FIRST50" if has_replay else "AUTHORS_NO_REPLAY"
    baseline=replace(
        baseline,model="fedclass_resnet8",
        method=("replay" if has_replay else "sgd"),
        memory_per_class=(50 if has_replay else 0),
        replay_reservoir_fraction=0.,
        replay_ratio=0., # Source CE uses concatenation, not 50/50 mixed replay.
        optimizer_state_policy="reset", # Original CE._train recreates SGD.
        aggregator="fedavg",classwise_head_merge=False,
        client_training_concurrency=client_training_concurrency,
        job_id=f"FEDCLASS_CE_{tag}_SOURCE_V2_FULL_SEED{seed}")
    rotating=replace(baseline,participation=.8,
                     stream_offline_policy="rotate_once",
                     job_id=f"FEDCLASS_CE_{tag}_SOURCE_V2_ROTATING_OFFLINE_SEED{seed}")
    for cfg in (baseline,rotating):cfg.validate()
    # The treatment cannot silently change anything beyond participation.
    a,b=baseline.to_dict(),rotating.to_dict()
    different={k for k in a if a[k]!=b[k]}
    if different!={"participation","stream_offline_policy","job_id"}:
        raise RuntimeError(f"Unmatched experimental settings: {different}")
    return baseline,rotating


def _output_root(args):
    tag=(f"seed{args.seed}_lr{args.lr:g}_epochs{args.epochs}"
         +"_source_v2")
    return Path(args.output_dir)/args.memory_mode/tag


def make_design(jobs,*,mode,replay_ratio):
    plan=audited_plan(jobs)
    plan.update(
        suite="FedCLASS",algorithm="FedCLASS CE + FedAvg",
        upstream_commit=SOURCE_COMMIT,
        upstream_ce="FedCLASS/models/ce.py",
        upstream_aggregation="FedCLASS/utils/fed_algorithm.py:FedAvg",
        original_backbone="FedCLASS/convs/resnet_client.py:resnet8",
        original_classifier="FedCLASS/convs/linears.py:SimpleLinear",
        adapted_model="fedclass_resnet8",
        local_loss="original FedCLASS/models/ce.py:CE._train, compact all-locally-known CE",
        client_class_history="last-synchronized global classes plus privately acquired classes",
        memory_mode=mode,
        replay_ratio=None, # Not an original-source hyperparameter.
        original_first_k_per_class=(50 if mode=="authors_exemplars" else 0),
        per_class_reservoir_capacity=0,
        replay_policy=("AUTHORS original BaseLearner._construct_exemplar_unified first-k per class; archives synchronized under experimental protocol"
                       if mode=="authors_exemplars" else "disabled"),
        upstream_fidelity="FedCLASS original ResNet8 + SimpleLinear + CE/FedAvg; v16 asynchronous class-IL adaptation",
        aggregation_on_return="v18 noncanonical G+alpha*(local-anchor), alpha=1/(1+version lag), BN-safe, single-stale terminal-flush cap 1/num_clients",
        fully_connected="5/5 clients upload and receive every synchronization",
        round_robin="4/5 clients upload and receive, one excluded but still trains",
        full_and_offline_same_gpu=0,
        concurrent_processes=2,
        five_clients_learning_concurrently_per_arm=True,
        test_oracle_task_id=False,
        native_source_unmodified=True,
        paper_algorithm_warning="Now executes original CE._train, original first-k exemplar selection, and original FedAvg.update, but class mapping, fixed physical batch64, 5 persistent independent client streams, shared memory and delayed upload are protocol adaptations; not original full fl_trainer orchestration.")
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(
        description="FedCLASS CE/FedAvg: concurrent all-seen-updates versus offline round-robin CIFAR-100 pilots")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--lr",type=float,default=.03)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--client-training-concurrency",type=int,default=0,
                   help="max active clients per simulation process; 0 means all five")
    p.add_argument("--memory-mode",choices=MEMORY_MODES,default="authors_exemplars",
                   help="authors_exemplars=original first-50/class source selection; none=source CE without exemplars")
    p.add_argument("--replay-ratio",type=float,default=1.)
    p.add_argument("--gpu",type=int,default=0,
                   help="Physical GPU 0 is mandatory for both concurrent workers")
    p.add_argument("--data-root",default="data")
    p.add_argument("--output-dir",default="results/fedclass_pilot")
    p.add_argument("--plan-only",action="store_true")
    p.add_argument("--permit-adapted-legacy",action="store_true",
                   help="Explicitly run the OLD CE/FedAvg adapted trainer, NOT the original CE._train path")
    p.add_argument("--no-download",action="store_true")
    p.add_argument("--worker",type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.gpu!=0:
        p.error("The concurrent FedCLASS pilots must both use GPU 0")
    jobs=make_fedclass_jobs(seed=args.seed,lr=args.lr,epochs=args.epochs,
                            memory_mode=args.memory_mode,replay_ratio=args.replay_ratio,
                            client_training_concurrency=args.client_training_concurrency)
    root=_output_root(args)
    root.mkdir(exist_ok=True,parents=True)
    design=make_design(jobs,mode=args.memory_mode,replay_ratio=args.replay_ratio)
    design['upstream_fidelity']=fidelity_report('fedclass')
    design['source_fidelity']='UNVERIFIED_PARTIAL_ADAPTATION'
    atomic_json(root/"design.json",design)
    if args.plan_only:
        print(json.dumps(design,indent=2),flush=True)
        return
    require_verified('fedclass',permit_legacy=args.permit_adapted_legacy)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 0 required; the production pilot never silently falls back to CPU")
    if args.worker is not None:
        result=run_experiment(jobs[args.worker],str(root/"runs"),args.data_root,download=False)
        print(json.dumps(result,indent=2),flush=True)
        return
    # Download/cache once before the two subprocesses access immutable CIFAR-100.
    from .data import get_data_bank
    get_data_bank("cifar100",args.data_root,download=not args.no_download,seed=args.seed)
    command=[
        sys.executable,"-m","cafcl.fedclass_primary",
        "--seed",str(args.seed),"--lr",str(args.lr),"--epochs",str(args.epochs),
        "--memory-mode",args.memory_mode,"--replay-ratio",str(args.replay_ratio),
        "--client-training-concurrency",str(args.client_training_concurrency),
        "--gpu","0","--data-root",args.data_root,"--output-dir",args.output_dir,
        "--no-download"]
    if args.permit_adapted_legacy:command.append('--permit-adapted-legacy')
    processes=[]
    codes=[]
    try:
        for i,job in enumerate(jobs):
            env=os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"]="0"
            env.setdefault("OMP_NUM_THREADS","1")
            env.setdefault("MKL_NUM_THREADS","1")
            log=root/f"job_{i}.log"
            handle=log.open("a")
            try:
                proc=subprocess.Popen(command+["--worker",str(i)],env=env,
                                      stdout=handle,stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                raise
            processes.append((proc,handle))
            print(f"Launched {job.job_id}: pid={proc.pid} GPU=0 log={log}",flush=True)
        while any(proc.poll() is None for proc,_ in processes):
            failed=[proc for proc,_ in processes if proc.poll() not in (None,0)]
            if failed:
                for sibling,_ in processes:
                    if sibling.poll() is None:sibling.terminate()
                for sibling,_ in processes:
                    if sibling.poll() is None:
                        try:sibling.wait(timeout=10)
                        except subprocess.TimeoutExpired:sibling.kill();sibling.wait()
                break
            time.sleep(.25)
        codes=[proc.wait() for proc,_ in processes]
    finally:
        for proc,handle in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            handle.close()
    if any(c!=0 for c in codes):
        raise RuntimeError(f"FedCLASS paired pilot failed {codes}; inspect {root}/job_*.log")
    comparison=build_report(root,jobs)
    if comparison is None:
        raise RuntimeError("A worker returned success without both final summaries")
    report=root/"comparison"/"comparison.md"
    narrative=report.read_text()
    label=("FedCLASS source CE + authors original first-50/class exemplars + original FedAvg"
           if args.memory_mode=="authors_exemplars"
           else "FedCLASS vanilla CE + FedAvg (zero replay)")
    narrative=narrative.replace("Two matched Replay/FedAvg runs,",f"Two matched {label} runs,")
    narrative+=(
        "\n**Source and method fidelity:** upstream FedCLASS ResNet8, SimpleLinear,"
        " CE and sample-weighted FedAvg; adapted to the v16 class-pair stream and"
        " offline-only communication intervention. Returning stale clients use"
        " a lag-damped, BN-safe stale rebase with a sole-stale terminal cap. See FedCLASS/pilot/README.md.\n")
    report.write_text(narrative)
    print("FEDCLASS_PAIRED_RESULT",json.dumps(comparison,indent=2),flush=True)
    print(f"Comparison: {report}",flush=True)


if __name__=="__main__":
    main()
