# Correlation-Aware Federated Continual Learning (CA-FCL)

A modular, research-oriented PyTorch simulator for **federated continual learning under partial client participation, class overlap, old/new class relationships, cross-client temporal recurrence, and local forgetting prevention**.

> **Implementation status (2026-11-17):** working, expanded research prototype—not a validated reproduction of all 152 planned research blocks. It registers **276 independently configured CIFAR-10 jobs and 278 CIFAR-100 jobs**, covering **63 distinct research block IDs** (some as partial/approximate implementations). The entire 152-block research catalogue is documented. **35 offline tests pass**. The actual CIFAR sweeps have **not** been trained or benchmarked here. Distinguish programmed jobs from completed research findings.

## Install

```bash
python -m pip install -e '.[test]'
pytest -q
```

Requires Python >=3.10, PyTorch, torchvision, SciPy, NumPy, and PyYAML. CUDA is chosen automatically when available. CIFAR datasets download through torchvision only when explicitly enabled.

## Run every registered executable job, one after another, batch size 512

```bash
# Inspect jobs first (no training, no download):
bash scripts/run_all_cifar10.sh --plan-only
bash scripts/run_all_cifar100.sh --plan-only

# Sequentially train and save all registered CIFAR-10 jobs:
bash scripts/run_all_cifar10.sh

# Sequentially train and save all registered CIFAR-100 jobs:
bash scripts/run_all_cifar100.sh

# Across datasets, without concurrent execution:
bash scripts/run_both_sequentially.sh
```

Use `--limit 1` to smoke-test the first job, `--stop-on-error` to abort on the first failure, and `--seed N` plus a **different** `RESULTS_ROOT` for independent replication sets. Use `DATA_ROOT` for the CIFAR download location. The scripts **never start automatically** after install or push.

The default plan on each dataset comprises the 60-job canonical full/partial-participation matrix; supported participation, overlap, relatedness, recurrence, memory, synchronization, aggregation, optimizer, selection, and diagnostic ablations; and expanded mechanisms including DER/DER++, ER-ACE, MAS, LwF, server FedAdam/FedAvgM, adaptive participation scheduling, heterogeneous local clients, retrospective leave-one-client-out aggregation, varying inactive-data retention, and **matched multi-round include/exclude counterfactual forks** from shared full-state checkpoints. See `configs/plans/` and `docs/EXPERIMENT_CATALOGUE.md`.

## Non-negotiable scientific semantics

1. **Data arrives at all clients**, irrespective of selection. This immediately expands the global evaluation population.
2. **Every selected client receives the current global model before local training**, including rejoining clients.
3. Clients train only on accessible local training indices and permitted private memory; no global raw training-data pooling is performed for optimization.
4. The server aggregates accepted local updates using the configured algorithm.
5. **Every selected active client immediately receives the new global model after aggregation.** A separate `full_sync` mode broadcasts to inactive clients too.
6. The official **global held-out evaluation includes every class observed anywhere so far**, even when held only by an inactive client. Global model exposure and never-integrated classes are separately recorded.
7. The local memory survives global synchronization. GEM/A-GEM constrain against locally remembered historical classes, which does not guarantee globally forgetting-safe aggregation.
8. Every accepted model update is versioned; completed jobs retain independent configurations, metrics, stage checkpoints, round-level fork checkpoints when requested, and a final model. An interruption can resume from a durable round checkpoint.

## Local algorithms

Implemented: SGD, local experience replay, A-GEM gradient projection, GEM quadratic-program projection with explicit residual checks, simplified diagonal-Fisher EWC, ER-ACE-like asymmetric stream masking and replay, local stage-consolidated DER and DER++, MAS-style local importance, LwF-style distillation, and optional FedProx local proximal regularization.

**Methodology qualifications:** ER-ACE, DER/DER++, EWC, MAS and LwF are *research adaptations* and are **not claimed as exact original-paper reproductions**. DER logits are consolidated at local-training boundaries rather than each incoming online stream step. GEM uses class-grouped client history and SciPy's CPU-based dual solve. Results must carry these qualifications.

## Selection / aggregation

Select: random, fixed, round-robin, least-recently-active, label coverage, old/new-class coverage, balanced, diversity, semantic-relatedness proxy, overlap, recurrence, fairness, stochastic-relatedness, and historical coverage. Selection based on simulator client labels is a **privileged oracle baseline**; not a privacy-preserving deployable solution.

Aggregate: sample-weighted FedAvg, uniform client averaging, FedAvgM-style server momentum, FedAdam-style adaptive server optimization. The latter two maintain server optimizer state and record their hyperparameters. Proximal stabilization (`fedprox_mu`) is implemented in local training, not as an aggregation operator.

## Correlation and causal analysis

The simulator supports disjoint/IID/Dirichlet-skewed, overlapping, related/unrelated class-order, temporal recurrence (same versus different client, delay), mixed, and asynchronous arrival scenarios. CIFAR-100 relationships use official coarse-superclass metadata; CIFAR-10 relationships use documented heuristic class pairings. Neither is ground-truth gradient similarity.

Opt-in diagnostics compute class-label Jaccard, feature-mean cosine, held-training-sample class-gradient cosine, temporal arrival correlations, and **retrospective aggregation-only leave-one-client-out effects**. These diagnostics have privileged access to simulator training statistics and cannot be advertised as deployable under strong federated privacy constraints. Retrospective aggregation counterfactuals are distinct from the implemented **limited multi-round checkpoint forks**. The latter resume two client-selection intervention arms from a shared full server/client/RNG snapshot; they do not exhaust the causal intervention search space.

## Run artifacts

```text
results/cifar10/
  job_plan.json
  suite_status.json
  runs/<INDEPENDENT_JOB_ID>/
    config.json
    scenario.pt
    status.json
    metrics.json
    checkpoint_latest.pt
    model_stage_01.pt
    model_stage_02.pt
    ...
    model_final.pt
    summary.json
```

The CIFAR-100 tree is identical. Every run has its own model and results. Stage and latest full-state checkpoints preserve client memory, model versions, participation RNG state, server optimizer state, and global metrics. Completed jobs are skipped only if the expected artifacts and matching configuration are present. Changed configurations cannot silently overwrite completed jobs.

The test-bank size expands when new labels arrive, so the raw full-population accuracy change is *not identical* to forgetting. Per-class longitudinal retention, previously learned/integrated status and old/new accuracy are reported separately. Classes never globally trained remain in global accuracy.

## Validate and inspect

```bash
pytest -q
python -m cafcl.cli catalogue --output catalogue.json
PYTHONPATH=src python -m cafcl.cli suite --dataset synthetic --suite core --limit 1
```

The smoke test uses an offline ten-class synthetic image dataset and is not evidence about CIFAR training performance.

## Documentation

- [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) — architecture, round protocol, algorithms, failure handling and causal interpretation.
- [`docs/EXPERIMENT_CATALOGUE.md`](docs/EXPERIMENT_CATALOGUE.md) — all 152 planned blocks, actual job coverage, and remaining scope.
- [`docs/STATUS.md`](docs/STATUS.md) — verified implementation limits, testing and open work.
- [`configs/plans/`](configs/plans/) — materialized batch-512 CIFAR experiment manifests.

## Limitations

This implementation **does not** complete every planned methodology. Broader fully controlled causal forks and all higher-order interactions, secure aggregation/differential privacy, robust federated deployment, SCAFFOLD, all-original-baseline reproduction, fine-grained controlled gradient-sign generation, full supervised task-free change-point learning, open-world label discovery, statistically significant CIFAR results, and per-client domain-shift evaluation remain future tasks. The catalogue reports missing blocks instead of turning them into fake SGD jobs. If a paper is written, report actual selected arms, realized data correlations, repetitions and confidence intervals, never the aspirational block count.
