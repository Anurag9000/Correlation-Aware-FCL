# CIFAR-100 v5 factorial ledger — feasibility-filtered, native 32×32 backbones

## Source and version

`src/cafcl/full_matrix.py` lazily enumerates the declared finite experiment grids. `src/cafcl/stream_audit.py` independently checks, for every generated configuration, all local-task label sets, canonical class coverage, within/across client-and-time overlap provenance, controlled outages, and finite class-example budgets. Misnamed/no-op overlap geometries are **excluded**, not silently treated as real interventions. Source to reproduce the audit: `python scripts/audit_stream_math.py`. The exact frozen hash, counts and exclusion report live in `configs/designs/stream_v5_native32.json`.

All CIFAR-100 production configurations now use either `vit_tiny_patch16_32` (`--backbone vit`, **default**) or `resnet18_cifar32` (`--backbone cnn`), 32×32 input and effective batch size 1024. The two backbones are **alternative executions of the same finite design**, not simultaneous model assignments in a single federated job. They have separate job IDs, checkpoints and output roots. The old CIFAR-10 and 20-class jobs remain retired.

## Mathematical audit

| Result | Jobs per backbone |
|---|---:|
| Old raw distinct generator registrations | 53,805 |
| Removed (impossible/absent/misnamed overlap mechanism) | **2,178** |
| **Feasible registered jobs retained** | **51,627** |
| Local methods × server methods | **11 × 9 = 99/99** |
| Balanced methodology core retained | **5,346** |

The 2,178 removed registrations are all in `task_length` configurations where the requested overlap was impossible or the combined condition did not actually instantiate every named geometry. For example, `intra_same_round` cannot occur with one task per client per communication interval, and cross-round overlap cannot occur when the entire novel stream ends after just one interval. The scheduler was also repaired to **never fill an incomplete final task using unrelated historical labels**; it may defer a terminal task to an eligible client slot or reject the schedule.

## Retained family counts (per backbone, after deduplication and feasibility audit)

| Family | Jobs | Varied controls |
|---|---:|---|
| Balanced method × server × overlap × participation × seed | 5,346 | 11 × 9 × 6 geometries × 3 rates × 3 seeds |
| Local task length | **6,732** | Classes/task 2,4; tasks/interval 1,2,3; invalid geometry cases removed |
| Offline duration | 13,662 | Consecutive disconnection 1–4 feasible intervals |
| Overlap strength | 2,970 | Probabilistic class reuse 25%, 50%, 100% |
| Raw sample overlap | 1,188 | Different-client shared-image fraction |
| Client selection | 10,692 | Selection policies and participation |
| Local SGD optimization | 3,267 | Learning rates × epochs |
| CL memory capacity | 756 | Examples per class |
| Importance regularization | 324 | EWC/SI/MAS/LwF strength |
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
| **Total** | **51,627** | Finite-grid jobs, not GPU executions |

The balanced core's six overlap geometries are: no overlap; same-client same-interval; same-client cross-interval; different-client same-interval; different-client cross-interval; combined. Its rates are 100%, 80%, 50%, with three seeds. The primary two-job probe intentionally does **not** use three seeds and does **not** have overlap.

## Relevant run modes

```bash
# default ViT-Tiny/16, native 32×32, batch1024
bash scripts/run_all_cifar100.sh --backbone vit --plan-only
# switch to native CIFAR-stem ResNet-18
bash scripts/run_all_cifar100.sh --backbone cnn --plan-only
# export exact finite-grid configurations
bash scripts/run_all_cifar100.sh --backbone vit --plan-only --export-plan outputs/cifar100_vit_v5.jsonl.gz
# validate both mathematical registry digests
python scripts/validate_factorial_plans.py
# primary intended first study: two jobs in parallel, one seed
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,1
```

**Scope:** This is exhaustive only over the *specified compatible finite axes*, not a full Cartesian product of all continuous hyperparameters or a claim that every source-coded variant is an exact paper implementation. Several advanced client-selection policies require privileged simulator metadata. The official test split must not drive adaptive client selection or hyperparameter tuning; the two-job probe evaluates official test only at the end.
