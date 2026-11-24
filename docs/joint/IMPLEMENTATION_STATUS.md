# Joint spatiotemporal FCL: current implementation status

## Implemented and tested core

The joint v2 protocol reuses the existing Replay/SGD + FedAvg engine.
It separates local atomic task time from synchronization time, supports
both IL settings, explicit task/head policies, exact/partial spatial and
temporal overlap, variable widths/depths, stable allocations, causal
sample/head exposure, controlled participation, stale uploads, accepted
occurrence receipts and a communication-only flush. Principal joint
conditions retain cross-client overlap. Explicit non-federated controls
are separate.

The most recent verified green CI before the epoch-slicing and planner
commits was cf9722b, 29 tests and 331 subtests:
https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/45408045478
This includes pooled Task-IL and Class-IL synthetic execution, local-only
and centralized controls, cross-semantic replay, failure-injection resume
equivalence on a short synthetic schedule, exact overlap solving and
distribution/semantic unit tests. Later changes require their own CI
verification. No real CIFAR-100 or GPU results are claimed.

## Catalog and controls

- Foundational: 48 cells by default.
- A–R screening: exact/partial sharing, migration, recurrence, topology,
  order, replication, common/private classes, heterogeneous widths/depths,
  omission, outages, raw sample overlap, dynamic overlap and novelty.
- X1–X8: 434 nominal interaction cells.
- Short: two- and three-task diagnostic motifs and head controls.
- Extensions: integer overlap solver (2–12 clients), weighted sample
  partitioning, wider/deeper grids, split/merge/recombination,
  deterministic receiver-domain transforms, actual CIFAR-100 superclass
  related/unrelated tasks.
- Controls: no-replay, local-only, centralized sequential and simultaneous
  current-data access preserving original Task-IL heads.
- Intratask: 1/2/5/10 synchronization opportunities per parent task,
  10 total epochs, same raw task data/head, no private replay insertion
  until parent completion. Epoch-offset shuffle; replay/augmentation
  randomness may change across fragments.
- Smoke: eight one-epoch real-data diagnostic conditions (two clients,
  partial class coverage, both local transition types, depths 1 and 2).

The all suite excludes smoke (a separate diagnostic). The compiler reports
nominal, infeasible, aliased and unique executable conditions, using the
actual TRAIN label sequence and superclass map. Do not quote a fixed total
from the earlier v1 registry. Plans are stored once by SHA-256 and reused
by the two settings when identical; family selectors recognize aliases.

## Commands

    git pull origin main
    PYTHONPATH=src pytest -q tests/test_joint_*.py
    PYTHONPATH=src python -m cafcl.joint.runner plan --suite smoke --output runs/joint_smoke
    PYTHONPATH=src python -m cafcl.joint.runner validate --manifest runs/joint_smoke/manifest.json --no-download
    PYTHONPATH=src python -m cafcl.joint.runner run-suite --manifest runs/joint_smoke/manifest.json --output runs/joint_smoke_results --gpu 0 --local-evaluation
    PYTHONPATH=src python -m cafcl.joint.runner audit --manifest runs/joint_smoke/manifest.json --output runs/joint_smoke_results

After real-data smoke verification:
    PYTHONPATH=src python -m cafcl.joint.runner plan --suite all --output runs/joint_all
    PYTHONPATH=src python -m cafcl.joint.runner validate --manifest runs/joint_all/manifest.json --no-download

The controller is sequential with explicit client concurrency, not a
GPU resource broker. Source launch receipts, checkpoint state, atomic
interval traces, knowledge receipts, standard server reports and optional
private/pre/post/broadcast validation are saved. An independent auditor
checks the plan and occurrence ledger. The report command reconstructs
acquisition, signed retention, integration latency, prediction collapse,
class retention and trajectory/curriculum plots. Class-IL comparisons with
different competitor universes are explicitly excluded from pure local
retention estimates. Parameter-only candidate update geometry is available
in detailed local evaluation mode.

## Remaining and limitations

- Historical V25 Class-IL objective and exact environment reproduction
  remain unverified; current Class-IL is an explicit new baseline.
- The historical legacy CI fails before pytest because its factorial
  registry contains Task-IL DER/DER++ raw-reuse configurations rejected by
  the current legacy validator. This failure predated the joint extension
  (original HEAD 6c46d9e). The archived legacy restriction and frozen
  design were not altered.
- Epoch-sliced intra-task synchronization does not implement arbitrary
  minibatch-level joint synchronization.
- Pooled, skewed and replication comparisons are data-matched where
  declared; optimizer work/communication budgets are not universally
  matched. Exact augmentation/replay draw matching across all schedules
  is not established.
- Some A–R variants remain constructive screens rather than exhaustive
  constrained temporal optimization. Not all graph geometries or all
  factor combinations are supported. Constraint solver is bounded.
- Complete gradient/representation probes, per-example prediction
  histories, full transactional fault injection beyond the verified
  boundary resume case, multi-GPU resource-broker integration and
  multi-run statistical synthesis remain unfinished.
- Domain replay uses receiver-domain transformations, not original
  source-domain pixel preservation. Canonical evaluation and terminal
  domain-specific scores are separate.
- Real-data smoke, full GPU training and scientific replication must
  actually run before empirical claims can be made.

For chronological implementation notes see STATUS.md and commit history.

## Current-task versus historical-task reporting

Detailed runs now save standard Task-IL task-conditioned test scores after
communication, as diagnostic-only held-out inference. Analysis separates
current task occurrence average, current/new class recall, historical
noncurrent class recall, and peak drops after class/head integration. It
writes the full local historical accuracy matrix, grouping Class-IL
forgetting by identical competitor sets. Resource receipts account for
actual sampled-mixed/SGD current draws, padding, replay draws, optimization
microbatches, optimizer steps, model tensor and archive image downloads;
network overhead and archive upload bytes are explicitly excluded.

## Multi-GPU campaign execution

The broker subcommand launches at most one worker per GPU UUID, with host-local
advisory GPU leases and an output-root lease, explicit free-memory and disk
thresholds, no automatic CPU fallback, a clean source revision gate, serial
manifest/dataset preflight and independent artifact verification. Attempt logs
and receipts are retained. It does not kill unrelated jobs, estimate a GPU's
required memory, or promise protection from non-cooperating external processes.
The operator must measure workload memory and choose the threshold. Linux only.
Existing completed or interrupted run configurations are never silently replaced.

    PYTHONPATH=src python -m cafcl.joint.runner broker --manifest runs/joint_smoke/manifest.json --output runs/joint_smoke_results --gpus 0 1 --min-free-mib 16000 --local-evaluation --no-download

The 16000 MiB figure is an illustrative operator threshold, not a verified
measurement. No broker jobs were launched on actual GPUs by this change.
