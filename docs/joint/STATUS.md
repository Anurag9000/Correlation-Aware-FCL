# Joint spatiotemporal FCL implementation status

This opt-in extension preserves legacy configuration signatures and scheduler
dispatch. Protocol: joint_spatiotemporal_fcl_v1_task_il / _class_il.

Implemented:
- Deterministic atomic compiler, exact replication and common/private
  overlap, arbitrary explicit sequences, heterogeneous depths via explicit
  events, stable task/head identities, disjoint raw training allocation.
- Periodic or explicit communication boundaries, explicit upload cohorts
  and fixed-duration random outages, terminal flush.
- 48 foundational design cells, with data-matched depth grouping.
- Hash-verified persisted plan adapter to the existing task-barrier engine.
- Explicit shared linear Class-IL head and opt-in configuration path.
- Semantic, occurrence and client-semantic Task-IL head policies.
- Local visible replay source head filtering; no future raw sample reuse
  supported by this protocol.
- Optional local validation before/after each atomic task; existing server
  metrics, aggregation and replay mechanisms reused.
- Planner/worker CLI and unit/synthetic integration tests.

Run on a checked-out repository with its dependencies:
    PYTHONPATH=src python -m unittest discover -s tests -p 'test_joint_*.py'
    PYTHONPATH=src python -m cafcl.joint.runner plan --output runs/joint_plan
    PYTHONPATH=src python -m cafcl.joint.runner run --manifest runs/joint_plan/manifest.json --job-id J-H-S1-O0-L2-task_il --output runs/joint_results --local-evaluation

The existing shared-class competition / frozen-row path is used for the new
Class-IL Replay baseline. This is not a verified reproduction of historical
V25. Existing Task-IL schedule-membership oracle remains the evaluator.
Local validation uses the same evaluator and visible task heads; it is not yet
the separate standard per-semantic-task evaluator.

No unit tests, synthetic runs, real-data runs or GPU results are claimed
verified by the source commits alone. Do not launch the 48-condition grid
until the new tests pass and short real-data diagnostics are inspected.

Outstanding: automated family A–R expansion and full constraints, topology
solver, controlled raw-sample reuse and domain allocation, intra-task joint
batch synchronization, standard task-conditioned metrics, pre-upload and
post-broadcast diagnostics, transactional commit hardening, complete lifecycle
and knowledge-flow instrumentation, complete campaign controller/resource
broker, resume/regression validation and empirical experiments. Explicit
schedules can encode many family motifs but are not equivalent to automated
validated family-specific experiment generators.

## Expanded screening and atomic-time audit

The screening catalog contains constructive initial A–G and I–O, Q–R
motifs in both learning settings, with H in the foundational registry.
The screening planner reports rejected schedules and low sample support.
Family P is explicitly blocked. Sequence permutations use client/semantic
sample allocation to match each client's raw sample multiset. The runtime
now schedules true atomic coordinates (including idle client gaps) and
checks a server model/archive metadata digest before and after private
learning. It emits an interval-level JSON trace with atomic occurrences.

Run the screening planner:
    PYTHONPATH=src python -m cafcl.joint.runner plan --suite screening --output runs/joint_screening

These are first-pass constructive screens, not the full proposed A–R
factorial catalog. Coverage tails, planned novelty, and oracle omission
must be interpreted as declared interventions; do not infer causal
isolation where other variables change. No test execution is claimed yet.
