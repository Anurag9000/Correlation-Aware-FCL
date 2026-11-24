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

## Continuation: causal provenance (protocol v2)

The prior final v1 CI passed 16 tests and 230 subtests at c6af986:
https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/45403370323

The opt-in v2 runtime now records local source-head exposure and only copies
acknowledged archive source heads at actual broadcast. It adds pending and
accepted occurrence receipts and per-interval knowledge JSON. Cross-semantic
raw reuse can be explicitly enabled. V1 checkpoints cannot resume into v2.
Commit-level durability remains boundary-based; full failure-injection and
resume equivalence still require tests. No GPU claims are made.

## Exact overlap and sample-skew extensions

The extensions suite adds a bounded exact integer overlap solver (2–12
clients, full pairwise intersection matrix), per-class occurrence weighted
sample partitioning, wider/deeper joint grids, and explicit split/merge/
recombination motifs. Weighted partitioning preserves unique training
support, but changes per-task sample count and optimizer work. No claim
of matched compute or domain adaptation is made. The all suite includes
these new designs; use generated manifests for counts.

## Explicit non-federated controls

The controls suite registers matched no-replay, local-only and centralized
sequential arms. Local-only uses empty cohorts and no terminal upload;
centralized sequential uses one learner and a single final export, with
no shared archive during learning. These are clearly declared reference
controls, not principal joint FL conditions. Private test results are
saved separately from the untrained local-only server score. Detailed
validation includes pre-upload snapshots of idle and reconnecting clients.
Simultaneous-access pooled/multi-head training is not yet implemented.

## Independent knowledge and result diagnostics

The artifact auditor now verifies each occurrence against the planned
training event, upload sender, accepted receipt, pending ledger and actual
server version. The report command reconstructs signed acquisition and
retention, global integrated-class peak drop, prediction collapse metrics,
replication strata and integration latency; unmatched Class-IL competitor
universes are not treated as pure forgetting. Detailed diagnostics include
parameter-only applied-candidate cosine/cancellation (not a claim that
whole-model weighting equals task-head weighting). A failure-injection
resume test compares continuous and resumed global tensors and receipt
ledgers for a short synthetic cross-semantic/stale-client curriculum.

    PYTHONPATH=src python -m cafcl.joint.runner report --run-dir runs/joint_results/J-H-S1-O0-L2-task_il

## Simultaneous-access oracle

Pooled controls expose all local tasks within an interval at its start,
shuffle their fresh examples jointly, keep the original Task-IL heads and
route current examples by original sample ID. Replay uses only previous
interval memory; private memory updates after the pooled experience, not
between source tasks. It uses the same fresh sample assignments and epochs
as the sequential counterpart; padding and optimizer steps can differ and
are reported, not claimed compute-matched. Occurrence rows share the pooled
before/after measurement, and group-level optimization work is recorded
once. Raw sample reuse is disallowed in this oracle to preserve unambiguous
fresh-task routing. Both IL settings are included in the controls suite.

## Distributional and semantic controls

Extensions now include fixed client domains, periodic receiver-domain
shifts and deterministic client-specific corruption. Raw images are
transformed after archive retrieval; replay is in the receiving client's
active domain. Canonical held-out metrics remain comparable; additional
terminal global-model domain-specific test metrics are stored separately.
Semantic same/different-superclass treatments use actual CIFAR-100 metadata
and match class occurrence counts, client counts, widths and horizon.

## Epoch-sliced intra-task synchronization

The intratask suite compares 1, 2, 5 and 10 communications per local task at
10 total epochs per parent task. Each fragment reuses its parent's exact
fresh sample IDs and classifier head; it does not increment the parent
local-experience counter or insert private replay until the last fragment.
The immutable plan preserves the parent hash and explicit fragment metadata;
receipts and audits distinguish fragment integration from completion of a
parent task. Current minibatch shuffle includes the epoch offset. Replay
and augmentation RNG streams are reinitialized per fragment, so exact
augmentation/replay draws are not claimed matched across synchronization
frequencies. Full participation only; arbitrary minibatch-level joint
synchronization remains separate unfinished work.
