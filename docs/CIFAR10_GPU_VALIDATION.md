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
is the registered baseline. Candidate outputs are independent under
`results/cifar10_full_validation_v2/runs/`; selection and artifact verification
are not complete until the full plan finishes. A candidate must exceed the
10-class chance accuracy separately on old and new classes over the final-stage
selection window before it can be ranked by harmonic mean.

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
branches. A final review must confirm the frozen hyperparameters are matched
across arms intended to isolate a different factor.

## Production boundary

The repository has 423 registered CIFAR-10 jobs across 144 scheduled research
blocks. The 17 representative jobs above do not complete that suite and do not
claim exhaustive implementation of all 152 catalogue blocks. After the
validation gates pass, run the full suite sequentially with `--stop-on-error`,
`--wait-for-exclusive-gpu`, and `--minimum-memory-gib 8`; then independently
verify all 423 job artifacts and produce the required analyses and reports with
`python scripts/analyze_cifar10_production.py --results-root results/cifar10`.
The analyzer exports registries, per-round and per-class CSVs, causal and
runtime/communication tables, and per-job curves and heatmaps under
`results/cifar10/analysis/`.
