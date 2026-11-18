"""Five-run CIFAR-10 A-GEM/FedAvg pilot with exact 25% CLIENT exclusion.

Four disjoint clients own randomized 3/3/2/2 label partitions. The baseline
includes all four; four arms permanently exclude one different client each.
Exactly one quarter of clients are absent, but 20% or 30% of labels are absent
according to the excluded owner. Four labels per client cannot be arranged
without overlap on a ten-class dataset.

The official ten-class global-evaluation invariant is preserved in the simulator.
Paired *research comparisons* exclude the two omitted labels from BOTH arms.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig
from .data import compile_scenario, get_data_bank
from .evaluation import evaluate
from .models import create_model
from .simulator import atomic_json, run_experiment

CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
CLIENTS = 4
CLASS_IDS = tuple(range(10))


def make_ledger(*, seed=0, quick=False, rounds_per_stage=None, memory_per_class=20,
                lr=0.03, model=None, train_fraction=None):
    """Pure deterministic design. Every omitted label appears in exactly one arm.

    Each client holds one stage-0 class (0..4), and one stage-1 class (5..9).
    Data allocation and model initialization share the same seed in every arm.
    """
    if rounds_per_stage is None:
        rounds_per_stage = 2 if quick else 20
    if model is None:
        model = 'linear' if quick else 'smallcnn'
    if train_fraction is None:
        train_fraction = 0.1 if quick else 1.0
    if not (0 < train_fraction <= 1):
        raise ValueError('train_fraction must be in (0,1]')
    if not (1 <= rounds_per_stage <= 10000):
        raise ValueError('rounds_per_stage must be positive')
    if memory_per_class < 1:
        raise ValueError('A-GEM pilot requires positive historical memory')
    rng = np.random.default_rng(seed + 20261118)
    early = tuple(map(int,rng.permutation(range(5))))
    late = tuple(map(int,rng.permutation(range(5,10))))
    groups = ((early[0],early[1],late[0]),
              (early[2],late[1],late[2]),
              (early[3],late[3]),
              (early[4],late[4]))
    holders = {str(cls):(cid,) for cid,group in enumerate(groups) for cls in group}
    assert sorted(cls for group in groups for cls in group)==list(CLASS_IDS)
    common = RunConfig(
        dataset='cifar10', num_clients=CLIENTS, stages=2,
        rounds_per_stage=rounds_per_stage, epochs=1, batch_size=512,
        method='agem', aggregator='fedavg', scenario='disjoint',
        selector='random', participation=1.0,
        holders_override=holders, memory_per_class=memory_per_class,
        class_sample_fraction=({str(c): train_fraction for c in CLASS_IDS}
                               if train_fraction < 1.0 else {}),
        seed=seed, model=model, lr=lr, full_sync=False,
    )
    mode = 'quick' if quick else 'production'
    plans = []
    baseline = replace(common, job_id='P00_full_participation', block_id='P00')
    baseline.validate()
    plans.append(dict(position=1, kind='baseline', off_client=None,
                      off_classes=[], included_classes=list(CLASS_IDS),
                      job_id=baseline.job_id, config=json.loads(json.dumps(baseline.to_dict()))))
    for k, pair in enumerate(groups):
        cfg = replace(common, job_id=f'P{k+1:02d}_omit_client_{k}', block_id='P01',
                      participation=0.75, target_client=k, force_target='exclude',
                      intervention_round=-1)
        cfg.validate()
        included = sorted(set(CLASS_IDS) - set(pair))
        plans.append(dict(position=k+2, kind='omit_fixed_client', off_client=k,
                          off_classes=list(pair), included_classes=included,
                          job_id=cfg.job_id, config=json.loads(json.dumps(cfg.to_dict()))))
    ledger = {
        'design_id': 'primary_cifar10_agem_fedavg_disjoint_25pct_clients_v2',
        'mode': mode, 'seed': seed, 'batch_size': 512,
        'method': 'agem', 'aggregation': 'fedavg',
        'note': 'Four unique omissions: exactly one of four clients off (25%) per arm. Class omissions are 30%,30%,20%,20% because ten labels cannot be split evenly among four disjoint owners.',
        'randomness': 'Randomly permute first-stage and second-stage class assignments ONCE, shared by all five jobs. Each arm permanently omits one distinct client.',
        'selection': 'Baseline all four clients each round; partial arms the other three clients each round (75% participation, 25% off). Fixed exclusion, not random dropout each round.',
        'data_partition': 'No cross-client class or raw-sample overlap; clients own randomized 3/3/2/2 classes. Some stages contain singleton class arrivals, so strict two-pair-per-client learning is mathematically impossible here.',
        'stage0_classes': list(range(5)), 'stage1_classes': list(range(5, 10)),
        'client_class_groups': {str(i): list(p) for i, p in enumerate(groups)},
        'class_names': list(CLASS_NAMES),
        'final_comparison': 'For each partial arm, evaluate BOTH saved final models on precisely its identical 7- or 8-class official-test subset. Never compare withheld labels.',
        'global_evaluation': 'Simulator still evaluates all arrived classes, including permanently inactive holders. Only the paired comparison excludes omitted labels.',
        'limitations': ['A 25% client reduction is confounded with removing 20% or 30% of class identities.',
                        'Single shared training seed is a diagnostic pilot, not a statistical causal estimate.',
                        'Exactly 25% of CLIENTS, not classes; ten labels cannot be divided into four equal integer sets.',
                        'Compatible paired episodes are used where two labels arrive together; singleton arrivals remain and do not satisfy the requested strict two-pair curriculum.',
                        'Validation/tuning should use training-derived validation only; this pilot does not tune hyperparameters.'],
        'jobs': plans,
    }
    serialized = json.dumps(ledger, sort_keys=True, separators=(',', ':'))
    ledger['design_sha256'] = hashlib.sha256(serialized.encode()).hexdigest()
    return ledger


def verify_partition(bank, cfg):
    """Verify a disjoint, stage-exposed 3/3/2/2 allocation."""
    s = compile_scenario(bank, cfg)
    for k in range(CLIENTS):
        expected = {int(c) for c, holder in cfg.holders_override.items() if tuple(holder) == (k,)}
        actual = {c for c, holders in s.label_holders.items() if k in holders}
        if actual != expected or len(actual) != len(expected) or len(expected)<2:
            raise AssertionError(f'Incorrect class holder {k}: {actual} != {expected}')
        stage0 = expected & set(range(5))
        stage1 = expected & set(range(5, 10))
        if not stage0 or not stage1:
            raise AssertionError('Every client must hold at least one earlier and one later class')
    if s.duplicate_sample_ids:
        raise AssertionError('Pilot may not duplicate training images across clients')
    if set(s.label_holders) != set(CLASS_IDS):
        raise AssertionError('Not all classes allocated')
    return s


def _load_final(run_dir, cfg, device):
    path = Path(run_dir) / 'model_final.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['configuration'] != cfg.to_dict():
        raise RuntimeError(f'Final model configuration mismatch in {run_dir}')
    model = create_model(cfg.model, 10).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return model


def _check_complete(root, job, cfg, expected_selected):
    path = Path(root) / 'runs' / job
    for fname in ('config.json', 'metrics.json', 'summary.json', 'status.json', 'model_final.pt'):
        if not (path / fname).exists():
            raise FileNotFoundError(path / fname)
    if json.loads((path / 'config.json').read_text()) != cfg.to_dict():
        raise ValueError(f'Unexpected configuration {job}')
    status = json.loads((path / 'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError(f'Incomplete job {job}: {status}')
    history = json.loads((path / 'metrics.json').read_text())
    total = cfg.stages * cfg.rounds_per_stage
    if len(history) != total:
        raise AssertionError(f'Incomplete rounds for {job}: {len(history)}/{total}')
    for rid, row in enumerate(history):
        if row['round'] != rid or sorted(row['selected_clients']) != sorted(expected_selected):
            raise AssertionError(f'Incorrect selected clients in {job}, round {rid}')
        if row['uploaded_count'] != len(expected_selected):
            raise AssertionError(f'Client did not upload as intended in {job}, round {rid}')
        if row['model_version'] != rid + 1:
            raise AssertionError(f'Missing aggregation in {job}, round {rid}')
    integrated = set(history[-1]['globally_integrated_classes'])
    expected_labels = {c for c in CLASS_IDS if cfg.holders_override[str(c)][0] in expected_selected}
    if integrated != expected_labels:
        raise AssertionError(f'Unexpected globally learned classes: {job}: {integrated} != {expected_labels}')
    return path


def compare_finished(ledger, root, data_root, download, allow_cpu=False):
    """Analyze exactly the same official-test class subset for both final models."""
    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for production model comparison; --allow-cpu for testing')
    device = torch.device('cuda' if torch.cuda.is_available() and not allow_cpu else 'cpu')
    bank = get_data_bank('cifar10' if ledger['mode']=='production' else ledger.get('dataset_override', 'cifar10'),
                         data_root, download=download, seed=ledger['seed'])
    run_root = Path(root)
    baseline_rec = ledger['jobs'][0]
    baseline_cfg = RunConfig(**baseline_rec['config'])
    baseline_dir = _check_complete(run_root, baseline_rec['job_id'], baseline_cfg,
                                   range(CLIENTS))
    baseline_model = _load_final(baseline_dir, baseline_cfg, device)
    baseline_all, _ = evaluate(baseline_model, bank, set(CLASS_IDS), 512, device, peak={})
    summary_rows=[]
    class_rows=[]
    for record in ledger['jobs'][1:]:
        cfg = RunConfig(**record['config'])
        active = tuple(i for i in range(CLIENTS) if i != record['off_client'])
        partial_dir = _check_complete(run_root, record['job_id'], cfg, active)
        partial_model = _load_final(partial_dir, cfg, device)
        seen = record['included_classes']
        baseline_seen, _ = evaluate(baseline_model, bank, set(seen), 512, device, peak={})
        partial_seen, _ = evaluate(partial_model, bank, set(seen), 512, device, peak={})
        row = {
            'partial_job': record['job_id'], 'off_client': record['off_client'],
            'off_classes': ','.join(map(str, record['off_classes'])),
            'off_class_names': ','.join(CLASS_NAMES[c] for c in record['off_classes']),
            'included_classes': ','.join(map(str,seen)),
            'baseline_all10_accuracy_pct':100 * baseline_all['accuracy'],
            'baseline_seen_accuracy_pct':100 * baseline_seen['accuracy'],
            'partial_seen_accuracy_pct':100 * partial_seen['accuracy'],
            'delta_seen_percentage_points':100 * (partial_seen['accuracy']-baseline_seen['accuracy']),
            'baseline_seen_macro_pct':100 * baseline_seen['macro_accuracy'],
            'partial_seen_macro_pct':100 * partial_seen['macro_accuracy'],
            'delta_macro_percentage_points':100 * (partial_seen['macro_accuracy']-baseline_seen['macro_accuracy']),
            'paired_test_samples':baseline_seen['test_samples'],
        }
        if baseline_seen['test_samples'] != partial_seen['test_samples']:
            raise AssertionError('Baseline and partial evaluated on different test populations')
        summary_rows.append(row)
        for cls in seen:
            class_rows.append({'partial_job':record['job_id'], 'off_client':record['off_client'],
                               'off_classes':','.join(map(str,record['off_classes'])),
                               'evaluated_class':cls,'evaluated_class_name':CLASS_NAMES[cls],
                               'baseline_accuracy_pct':100 * baseline_seen['per_class'][cls],
                               'partial_accuracy_pct':100 * partial_seen['per_class'][cls],
                               'delta_percentage_points':100 *(partial_seen['per_class'][cls] -baseline_seen['per_class'][cls])})
    reports=run_root/'reports';reports.mkdir(parents=True,exist_ok=True)
    for name,rows in [('paired_summary.csv',summary_rows),('seen_class_effects.csv',class_rows)]:
        with (reports/name).open('w',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(reports/'paired_summary.json',{
        'design_sha256':ledger['design_sha256'],
        'baseline_final_accuracy_all10_pct':100 * baseline_all['accuracy'],
        'summary':summary_rows,'per_class':class_rows,
        'comparable_labels_only':True,
        'interpretation':'Descriptive differences, not a pure causal effect of class identity or a population confidence interval.'})
    lines=['# Primary A-GEM/FedAvg class-omission pilot: paired final-server accuracy','',
           'Four non-repeating 25%-of-client omissions; compare the same **7 or 8 included official-test classes** in each arm.','',
           f'Full-participation baseline on all ten test classes: **{100*baseline_all["accuracy"]:.3f}%** (context only).','',
           '| Partial job | Withheld class labels | Baseline (same seen) | Partial (same seen) | Change (pp) |',
           '|---|---|---:|---:|---:|']
    for row in summary_rows:
        lines.append(f'| {row["partial_job"]} | {row["off_classes"]} ({row["off_class_names"]}) | '
                     f'{row["baseline_seen_accuracy_pct"]:.3f}% | {row["partial_seen_accuracy_pct"]:.3f}% | '
                     f'{row["delta_seen_percentage_points"]:+.3f} |')
    lines.extend(['', 'Withheld classes are never included in the paired comparison.',
                  'Negative delta = worse accuracy on *other, actually participated* classes.',
                  'The full model sees more training data and one additional contributor; this is a diagnostic comparison, not proof of a class-specific causal effect.',
                  '', 'Per-class differences for every included class: `seen_class_effects.csv`.'])
    (reports/'paired_comparison.md').write_text('\n'.join(lines)+'\n')
    return summary_rows


def main(argv=None):
    p=argparse.ArgumentParser(description='Five-run disjoint CIFAR-10 A-GEM/FedAvg 25%-of-clients omission pilot')
    p.add_argument('--output-dir',default='results/primary_agem_cifar10')
    p.add_argument('--data-root',default='data')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--quick',action='store_true',help='2 rounds/stage, 10% per-class train images, linear model (not production)')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--compare-only',action='store_true')
    p.add_argument('--limit',type=int,default=0,help='Run first N jobs for diagnostic testing only')
    p.add_argument('--rounds-per-stage',type=int,default=None)
    p.add_argument('--memory-per-class',type=int,default=20)
    p.add_argument('--lr',type=float,default=0.03)
    p.add_argument('--model',choices=['smallcnn','linear','mlp'],default=None)
    p.add_argument('--train-fraction',type=float,default=None)
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--allow-cpu',action='store_true',help='Testing only; production requires CUDA')
    args=p.parse_args(argv)
    if args.limit < 0:
        p.error('--limit cannot be negative')
    ledger=make_ledger(seed=args.seed,quick=args.quick,
                       rounds_per_stage=args.rounds_per_stage,
                       memory_per_class=args.memory_per_class,lr=args.lr,
                       model=args.model,train_fraction=args.train_fraction)
    mode=ledger['mode'];root=Path(args.output_dir)/mode
    if root.joinpath('ledger.json').exists():
        existing=json.loads((root/'ledger.json').read_text())
        if existing.get('design_sha256')!=ledger['design_sha256']:
            raise RuntimeError('Design changed for this output root: choose another output-dir to avoid mixing runs')
    else:
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'ledger.json',ledger)
    for job in ledger['jobs']:
        print(f'{job["job_id"]}: OFF={job["off_classes"]}, '
              f'included={job["included_classes"]}, participation={job["config"]["participation"]}')
    if args.plan_only:
        return ledger
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; use --allow-cpu only for offline debugging')
    if not args.compare_only:
        bank=get_data_bank('cifar10',args.data_root,download=not args.no_download,seed=args.seed)
        verify_partition(bank,RunConfig(**ledger['jobs'][0]['config']))
        for job in ledger['jobs'][:args.limit or None]:
            cfg=RunConfig(**job['config'])
            result=run_experiment(cfg,str(root/'runs'),args.data_root,
                                  download=not args.no_download)
            print(json.dumps(result,sort_keys=True),flush=True)
    if args.compare_only or not args.limit or args.limit==len(ledger['jobs']):
        summary=compare_finished(ledger,root,args.data_root,
                                 download=not args.no_download,allow_cpu=args.allow_cpu)
        print(f'Wrote {len(summary)} same-class final-model comparisons to {root}/reports')
    return ledger


if __name__=='__main__':
    main()
