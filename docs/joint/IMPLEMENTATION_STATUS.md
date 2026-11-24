# Joint spatiotemporal FCL: implementation and evidence boundary

## Implemented source

The existing Replay/SGD + FedAvg simulator remains the training engine.
The opt-in joint protocol does not rewrite historical experiment outputs.
It has deterministic atomic tasks, semantic and occurrence-specific heads,
explicit shared Class-IL heads, persisted plan hashes, fixed-draw training
sample allocation, communication regrouping, arbitrary explicit event
streams, controlled upload cohorts and terminal flush. Clients perform
their atomic tasks privately and server model/archive metadata hashes are
checked before/after each communication interval's local learning.

The foundational registry has 48 nominal cells. The constructive A–R
screening catalog has 208 design entries, crossed with the two IL settings
(416 nominal cells). It includes restricted same-semantic raw-sample overlap
for P. The eight X1–X8 interaction grids have 434 nominal cells, and the
short suite has 36 two-setting motif cells plus two Task-IL head controls.
The all-suite default therefore has 936 nominal entries before aliases and
infeasibility filtering. These are design counts, not verified executable
job counts. Plan and validate with the real TRAIN split to determine them.
Some arms intentionally use a coverage tail; do not treat all screens as
isolated causal interventions.

## Verified evidence

The independent joint-specific GitHub Actions run at commit
9070cb36d94259653478e22e210d10858481e633 passed 14 tests and 225 subtests.
This includes synthetic end-to-end Task-IL and Class-IL, atomic compiler,
screening catalog, task-conditioned evaluator, persisted plan roundtrip,
and nominal interaction/short-motif tests.
https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/45402962862

The subsequent restricted raw-reuse feature and final planner changes have
their own CI runs; do not conflate the earlier green run with validation
of later commits. No full CIFAR-100 or GPU experiment has been launched
by this implementation effort.

The existing legacy workflow fails before pytest on an unsupported
Task-IL DER/DER++ raw-sample-overlap configuration in the historical
factorial registry. The same failure occurred at original HEAD
6c46d9e1ea14a6aad4840436e8dfa2c8af3782b9. This was not introduced by the
joint implementation. The legacy scientific restriction remains in force.

## Commands

    git pull origin main
    PYTHONPATH=src pytest -q tests/test_joint_*.py

    PYTHONPATH=src python -m cafcl.joint.runner plan --suite short --output runs/joint_short
    PYTHONPATH=src python -m cafcl.joint.runner plan --suite foundational --output runs/joint_foundational
    PYTHONPATH=src python -m cafcl.joint.runner plan --suite all --output runs/joint_all
    PYTHONPATH=src python -m cafcl.joint.runner validate --manifest runs/joint_all/manifest.json --no-download

After verification, the short suite can be executed independently:

    PYTHONPATH=src python -m cafcl.joint.runner run-suite --manifest runs/joint_short/manifest.json --output runs/joint_short_results --gpu 0 --local-evaluation

The controller is sequential with explicit client concurrency, not a GPU
broker. It writes failure records instead of accepting incomplete runs.
Audit results with:

    PYTHONPATH=src python -m cafcl.joint.runner audit --manifest runs/joint_short/manifest.json --output runs/joint_short_results

## Remaining scientific and engineering requirements

This is not the complete 100-section design. Remaining work includes:
- Verified reproduction of historical V25 Class-IL, rather than merely the
  new explicit seen-class objective.
- Full arbitrary topology/recurrence constraint solving and every A–R
  variant, not just the constructive screen and explicit-event interface.
- Joint intra-task synchronization, all width/depth/participation extensions,
  heterogeneous domain transformations and distribution/quantity skew.
- Cross-semantic sample reuse with event-causal provenance. It is rejected
  rather than silently allowing future task-head access.
- Simultaneous-access local oracle, centralized and local-only controls,
  complete matched compute/communication budget comparisons.
- Full event-level knowledge-flow acknowledgements and transaction hardening,
  exact resume equivalence and expanded legacy regression tests.
- Full gradient/representation diagnostics, statistical comparison reports,
  curriculum plots, full resource-broker integration and research ledgers.
- Real-data smoke tests, full GPU campaign and statistical replications.

The available validation is source/unit/synthetic evidence only. It does not
establish high accuracy, causal identification, historical result equivalence
or publication-level novelty.
