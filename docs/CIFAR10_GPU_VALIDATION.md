# CIFAR-10 GPU validation record

This record separates the short hyperparameter screen, full-schedule validation,
representative mechanism checks, and the later 423-job production suite. Passing
one stage does not imply that the next stage has run.

## Short validation screen

`results/cifar10_tuning_v1/` contains eight paired SGD candidates using 50% of
eligible training data, eight rounds per experience, batch size 512, and
validation-only selection. The independent artifact verifier passed 8/8 CUDA
jobs. The highest short-run harmonic mean was for learning rate 0.01 and
momentum 0.9 (old 21.1%, new 15.3%). This result is only a finalist signal; the
short screen did not establish the production hyperparameters.

## Full-data finalist validation

`results/cifar10_full_validation_v2/job_plan.json` contains six paired SGD
candidates: learning rates 0.01, 0.03, and 0.1 crossed with momentum 0 and 0.9.
Each uses all eligible training data, two experiences, 20 rounds per
experience, batch size 512, and validation-only selection. The 0.03/0 candidate
preserves the original baseline configuration. All six completed and passed the independent
artifact verifier, including complete round histories, CUDA execution records,
finite metrics and final models, and successful CUDA reload/inference. Selection
uses the final four rounds, requires both old and new accuracy to exceed 10%,
then ranks by harmonic old/new accuracy, macro accuracy, coverage and forgetting.
The selected setting is `lr=0.1, momentum=0.0`:

| Candidate | Old validation | New validation | Harmonic mean | Macro accuracy | Forgetting | Chance gate |
|---|---:|---:|---:|---:|---:|---|
| `lr=0.1, momentum=0.0` | 19.69% | 36.79% | 25.65% | 28.24% | 0.4415 | Pass |
| `lr=0.01, momentum=0.0` | 19.77% | 17.36% | 18.49% | 18.57% | 0.2050 | Pass |
| `lr=0.01, momentum=0.9` | 16.80% | 13.37% | 14.89% | 15.09% | 0.6786 | Pass |
| `lr=0.03, momentum=0.9` | 18.36% | 11.57% | 14.19% | 14.97% | 0.7244 | Pass |
| `lr=0.1, momentum=0.9` | 12.35% | 14.15% | 13.19% | 13.25% | 0.6809 | Pass |
| `lr=0.03, momentum=0.0` | 39.04% | 6.09% | 10.54% | 22.57% | 0.0638 | Fail |

The selected candidate and detailed ranking are recorded in
`results/cifar10_full_validation_v2/selection.json` and `selection.csv`; the
independent verifier receipt is `verification.json`. Per-job GPU PIDs and
memory snapshots are in `execution_context.jsonl`. The v2 run overlapped the
existing CUDA workload with a 4 GiB free-VRAM gate and 5 GiB inter-job host-RAM
gate. Runtime is recorded but excluded from ranking because shared load affects
timing. Final official-test metrics are saved per candidate but are not used in
selection.

An earlier attempt is retained in `results/cifar10_full_validation_v1/` for
provenance. Its 0.01/0 candidate completed before another CUDA job started. Its
0.1/0 candidate overlapped a Drift-Aware-BGD CUDA process and is marked
potentially resource-contaminated in `execution_notes.json`; exclude it from
selection and runtime comparisons. The runner stopped between jobs at the RAM
gate. No output was deleted or overwritten.

The completed 0.01/0 v1 run recorded 40 CUDA rounds at batch 512, with 568.79
summed per-round seconds (14.22 seconds per round) in `metrics.json`. This is a
single-job timing observation, not a full-suite runtime guarantee; the
overlapping 0.1/0 candidate is excluded from timing estimates.

## Representative production configurations

After the finalist choice, validate these existing production job IDs with
their registered full-data, batch-512, 20-round-per-experience configurations.
They write to `results/cifar10/runs/`, the same artifact tree used by the final
suite, so completed valid jobs can be reused by the 423-job run.

| Requirement | Job IDs |
|---|---|
| Full-participation SGD baseline | `A01_disjoint_sgd_full` |
| Random 50% and 25% participation | `A01_disjoint_sgd_random50`, `A01_disjoint_sgd_random25` |
| Full and partial GEM | `A01_disjoint_gem_full`, `A01_disjoint_gem_random50` |
| Full and partial A-GEM | `A01_disjoint_agem_full`, `A01_disjoint_agem_random50` |
| Local replay | `G04_local_replay` |
| Related and unrelated classes with replay | `B11_extended_relation_related_local_replay`, `B11_extended_relation_unrelated_local_replay` |
| Within-round class overlap | `C02_overlap_sgd_random50` |
| Cross-client recurrence | `D07_recurrence_sgd_random50` |
| Related-client causal fork baseline and branches | `L02_forkbaseline_related_sgd`, `L02_forkinclude_related_sgd`, `L02_forkexclude_related_sgd` |
| Advanced aggregation | `E10_extended_relation_aggregation_fedadam` |
| Gradient-aware selection | `N04_closure_gem_informed_gem` |

The canonical plan order keeps each causal fork baseline before its dependent
branches. The tuned `lr=0.1` is shared across CIFAR-10 jobs, while the H04
momentum value 0.9 remains an explicit experimental factor in both matched arms.
The CIFAR-100 plan retains its existing `lr=0.03` baseline.

## Production boundary

The repository has 423 registered CIFAR-10 jobs across 144 scheduled research
blocks. The 17 representative jobs above do not complete that suite and do not
claim exhaustive implementation of all 152 catalogue blocks. After the
validation gates pass, run the full suite sequentially with `--stop-on-error`,
`--minimum-free-vram-gib 4`, and `--minimum-memory-gib 5`. The user authorized
concurrent CUDA work when VRAM is available; the suite still runs one CAFCL job
at a time. Per-job start and end VRAM, RAM, and CUDA PID snapshots are recorded
in `results/cifar10/execution_context.jsonl`. Then independently verify all 423
job artifacts and produce the required analyses and reports with
`python scripts/analyze_cifar10_production.py --results-root results/cifar10`.
The analyzer exports registries, per-round and per-class CSVs, causal and
runtime/communication tables, and per-job curves and heatmaps under
`results/cifar10/analysis/`.
