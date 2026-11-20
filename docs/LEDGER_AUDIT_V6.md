> **Historical pilot design:** The only current primary Replay/FedAvg recipe is [PRIMARY_PROBE.md](PRIMARY_PROBE.md). This v6 ledger is retained for historical feasibility math; its primary-probe description and commands are retired.

# CIFAR-100 native32 v6: exact experiment-ledger mathematics

This is a reproducible **source-configuration feasibility audit**, not evidence of GPU training or scientific accuracy.

## Primary pilot: exactly two jobs

Five clients each process two consecutive binary-class local tasks per communication interval, for five training intervals. Thus 5 clients × 2 tasks × 2 new classes × 5 intervals = **100 distinct CIFAR-100 classes**. They share the same image stream, seed 0, native ViT-Tiny/16 by default, effective batch 1024, class-aware accumulated-delta FedAvg, and local A-GEM. The baseline synchronizes 5/5 clients; the intervention synchronizes 4/5, hiding a different client each interval. Across five intervals each client is hidden exactly once. The sixth round is a **communication-only flush** to integrate the last offline update; no new local examples are trained. Neither run has engineered class overlap.

A 10-client disjoint schedule with 4 fresh classes/client/interval needs 3 training intervals (40 + 40 + 20); assuming 5 such full intervals would require 200 new class slots, which is impossible without reuse.

## Full ledger: what was checked

For each declared class task, the scheduler/auditor checks exactly the configured number of distinct active classes, global coverage of all labels 0–99, no future-label leakage or unrelated terminal-task fillers, correct occurrence/provenance for the named within/across-client and within/across-round overlap geometry, and a realizable per-label sample budget from CIFAR-100's 500 original training images/class. Repeated class occurrences share separate train samples unless raw overlap is specifically enabled.

For D-round offline interventions, every client outage must last **exactly D consecutive training communication intervals**, followed by a reconnection. A terminal synchronization is not itself an additional offline training interval. For instance, 3 training intervals with D=2 yield a truncated last cohort and are excluded.

The audit also excludes LwF memory-capacity and generic regularization-coefficient changes that its implementation does not consume. Separately, it prunes analytically inert selection interventions: different selection policies when all clients already participate; aliases `fairness`=`least_recent`, `diversity`=`coverage`, and `balanced`=`coverage` when old/new labels partition observed classes; plus `recurrence` and `overlap` selectors without recurrence or shared labels in the stream.

| Result | Jobs per selected backbone |
|---|---:|
| Original distinct registrations | 53,805 |
| Unrealized overlap geometry removed | 2,178 |
| Invalid exact offline durations removed | 6,534 |
| Inert/aliased selectors removed | 6,534 |
| Unused LwF memory-capacity settings | 108 |
| Unused LwF generic regularization settings | 81 |
| **Retained valid registrations** | **38,370** |
| Balanced factorial core preserved | 5,346 |
| Local × server pairs covered | 11 × 9 = 99 |

The exact SHA256 manifests for each backbone and per-reason job examples are stored in `configs/designs/stream_v6_native32.json`.

```bash
python scripts/audit_stream_math.py
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_all_cifar100.sh --backbone vit --plan-only
bash scripts/run_all_cifar100.sh --backbone cnn --plan-only
```

## Important scientific and computational caveats

Feasibility is **not** GPU memory assurance, convergence, method-paper fidelity, privacy, or statistical significance. Only official CIFAR-100 GPU results can establish research outcomes. Advanced metadata-driven selectors are simulator oracles, not deployable private-selection strategies.

At native 32×32, ViT-Tiny/16 sees only four 16×16 image patches. Each disjoint binary local task has 2 × 500 = 1,000 training images; with effective batch 1,024, one epoch produces **only one optimizer step per task**. Across five intervals, one epoch corresponds to just 10 optimizer steps per client. This is probably too little for a meaningful negative result from a randomly initialized ViT. For a more informative two-job pilot, use `--epochs 10` for *both* arms (still two jobs). Do not tune using the official held-out test split.

Logical batch 1024 uses default physical microbatches of 16; on ResNet-18, BatchNorm runs on those microbatches and is not mathematically equivalent to a physical batch of 1024. The v6 correction ensures A-GEM/GEM reference-gradient computation does not inadvertently update BatchNorm running statistics. The server merges accumulated stale deltas and class-specific classifier rows; it is not ordinary synchronous vanilla FedAvg.

## Start on mtechdeserver after pulling main

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone vit --plan-only
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpu 0 --epochs 10
```

Historical two-device wording is superseded: `--gpu 0` starts both federated runs concurrently on the SAME physical GPU. Each run concurrently trains all local client models at each task ordinal, then performs server aggregation. The launcher independently logs, checkpoints, and compares both runs. Source/CI checks do **not** claim remote GPU jobs were started.
