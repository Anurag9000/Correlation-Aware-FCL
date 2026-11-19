> **ACTIVE v14:** The full 11-method × 9-server-algorithm suite now defaults to effective batch 64 and momentum SGD 0.9/weight decay 5e-4, standard ViT-Base/16 (or CIFAR ResNet-18), first-four-then-two task cadence, and a separate global accuracy table after every server sync. Historical geometry and file paths below are retained for comparison. [Current v14 protocol](V14_SEEN_CLASS_OPTIMIZATION_AND_SYNC_EVAL.md).

# CIFAR-100 v6 factorial ledger — feasibility-filtered, native 32×32 backbones

## Source and version

`src/cafcl/full_matrix.py` lazily enumerates the declared finite experiment grids. `src/cafcl/stream_audit.py` independently checks, for every generated configuration, all local-task label sets, canonical class coverage, within/across client-and-time overlap provenance, controlled outages, and finite class-example budgets. Misnamed/no-op overlap geometries are **excluded**, not silently treated as real interventions. Source to reproduce the audit: `python scripts/audit_stream_math.py`. The exact frozen hash, counts and exclusion report live in `configs/designs/stream_v6_native32.json`.

All CIFAR-100 production configurations now use either `vit_tiny_patch16_32` (`--backbone vit`, **default**) or `resnet18_cifar32` (`--backbone cnn`), 32×32 input and effective batch size 1024. The two backbones are **alternative executions of the same finite design**, not simultaneous model assignments in a single federated job. They have separate job IDs, checkpoints and output roots. The old CIFAR-10 and 20-class jobs remain retired.

## Mathematical audit

| Result | Jobs per backbone |
|---|---:|
| Old raw distinct generator registrations | 53,805 |
| Rejected (unrealized overlap, truncated outage, ineffective selector, unused LwF controls) | **15,435** |
| **Feasible registered jobs retained** | **38,370** |
| Local methods × server methods | **11 × 9 = 99/99** |
| Balanced methodology core retained | **5,346** |

Rejections comprise **2,178** unrealized overlap geometries, **6,534** truncated exact-duration offline schedules, **6,534** no-op/alias selection variants, and **189** unused LwF hyperparameter variations. For example, `intra_same_round` is impossible with one task per client per interval, and a D=2 offline cohort started in the last of three training rounds cannot complete two hidden intervals. Selector policies with 100% participation select everyone, while some source implementations are exact aliases or have no signal on a nonrecurrent disjoint stream. The scheduler never fills an incomplete terminal task with unrelated labels. See [the v6 audit ledger](LEDGER_AUDIT_V6.md).

## Retained family counts (per backbone, after deduplication and feasibility audit)

| Family | Jobs | Varied controls |
|---|---:|---|
| Balanced method × server × overlap × participation × seed | 5,346 | 11 × 9 × 6 geometries × 3 rates × 3 seeds |
| Local task length | **6,732** | Classes/task 2,4; tasks/interval 1,2,3; invalid geometry cases removed |
| Offline duration | 7,128 | Consecutive disconnection 1–4 feasible intervals |
| Overlap strength | 2,970 | Probabilistic class reuse 25%, 50%, 100% |
| Raw sample overlap | 1,188 | Different-client shared-image fraction |
| Client selection | 4,158 | Selection policies and participation |
| Local SGD optimization | 3,267 | Learning rates × epochs |
| CL memory capacity | 648 | Examples per class; LwF no-op arms removed |
| Importance regularization | 243 | EWC/SI/MAS strength; LwF uses distillation strength instead |
| FedProx | 891 | Local proximal coefficient |
| Server optimization | 264 | FedAvgM/FedAdam server LR |
| Replay ratio | 324 | Replay-like local methods |
| Distillation strength | 243 | DER/DER++/LwF |
| Replay-loss coefficient | 162 | ER-ACE/DER++ |
| GEM grouping | 54 | Class/stage/all history constraints |
| Client population | 594 | 5, 10, 20 |
| Additional recurrent intervals | 1,782 | 0, 2, 4 explicit recurrence rounds |
| Semantic ordering | 594 | Natural, related, unrelated class order |
| Class imbalance | 891 | 0, .5, 1, 2 |
| Client quantity skew | 594 | 0, 1, 3 |
| Terminal flush | 297 | Enabled/disabled |
| **Total** | **38,370** | Finite-grid jobs, not GPU executions |

The balanced core's six overlap geometries are: no overlap; same-client same-interval; same-client cross-interval; different-client same-interval; different-client cross-interval; combined. Its rates are 100%, 80%, 50%, with three seeds. The primary two-job probe intentionally does **not** use three seeds and does **not** have overlap.

## Relevant run modes

```bash
# default ViT-Tiny/16, native 32×32, batch1024
bash scripts/run_all_cifar100.sh --backbone vit --plan-only
# switch to native CIFAR-stem ResNet-18
bash scripts/run_all_cifar100.sh --backbone cnn --plan-only
# export exact finite-grid configurations
bash scripts/run_all_cifar100.sh --backbone vit --plan-only --export-plan outputs/cifar100_vit_v6.jsonl.gz
# validate both mathematical registry digests
python scripts/validate_factorial_plans.py
# primary intended first study: two jobs in parallel, one seed
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpu 0
```

**Scope:** This is exhaustive only over the *specified compatible finite axes*, not a full Cartesian product of all continuous hyperparameters or a claim that every source-coded variant is an exact paper implementation. Several advanced client-selection policies require privileged simulator metadata. The official test split must not drive adaptive client selection or hyperparameter tuning; the two-job probe evaluates official test only at the end.


**v11 execution semantics:** All nine server algorithms are audited and classified in `SERVER_ALGORITHM_FIDELITY_V11.md`. The original 38,370-per-backbone factorial *geometry* is preserved, but earlier trained data/checkpoints do not constitute results of this new aggregation protocol. The v11 path prefix is `cifar100_native32_v11_server_fidelity_`. Canonical methods never apply the custom classwise row overwrite, even when a historical config record includes its legacy default flag.


**v13 factorial geometry change:** Only 2 classes per local task and 2 local tasks at round 0 are supported; later rounds always run 1 local task. Prior task-length ablation candidates requesting other dimensions are pruned. The active stream manifest, feasible-count and checksum live in a separately frozen v13 design (the v6 ledger remains historical). All 11 client modes × all nine server algorithms share this single schedule. See `FIRST4_THEN2_FULL_CL_V13.md`.


**v13 frozen counts (superseding the older v6/v12 numbers in historical sections):** 53,805 candidates; 32,133 valid jobs per backbone; 21,672 pruned. Nine training intervals for the 5-client pilot, with two initial local tasks followed by one per round. 99/99 client-method/server-method combinations retained. See `configs/designs/stream_v13_first4_then2.json` for exact family counts/checksums.
