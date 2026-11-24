# Second audit-remediation pass

This follow-up implements additional development tasks from the audit. It is
a code-change record, not an empirical result or an assertion of green CI.

## New runtime contracts

- Task-IL DER/DER++ targets are keyed by (TRAIN index, source head), with
  explicit source validation, first-target retention and head-local logits.
  Class-IL keeps index-keyed targets. This removes the legacy raw-reuse
  validation blocker; it does not certify every legacy algorithm adaptation.
- Principal and counterfactual Task-IL scores use independent task IDs.
  Historical class-membership-union tables remain explicitly labelled and
  separate. Class-IL reports one-step fixed-context validation retention.
- Feature panels use seeded class-wise sampling. EWC/MAS importance indexes
  the active task's local columns, not global class IDs.
- Campaign audits require launch/source/data receipts and per-sync content
  checksums, validate CSV counts against the JSON, and decode PNG integrity.
  This is not independent semantic reconstruction of every plotted pixel.
- Stationary resume is round-level; matched/reconstruction resume is
  parent-stage-level, including client models, optimizer slots, archives,
  pending data, RNG, history and evidence hashes. It does not restore midway
  through a partially completed stage.
- Independent joint Replay archive upload cohorts (selected/all) and
  download cohorts (selected/all/none) are separate from the model cohort.
  Archive-only receipts do not acknowledge model learning. This is still
  raw-data sharing, not privacy-preserving FL.
- Head weighting distinguishes historical exposure, measured current
  draws plus fractional replay exposure, and equal contributing clients.
  Logical payload accounting includes DER logits, importance tensors,
  teacher and original-method tensor artifacts; serialization, metadata
  and real network behavior are excluded.

## Explicit reference engines

The synchronous Dirichlet FCIL reference supports global tasks, sampled
online clients, local-optimizer reset, private replay and calibration
(NLL, Brier, ECE). It is not an exact reproduction of a specific paper.

```bash
PYTHONPATH=src python -m cafcl.reference_fcil plan --config configs/synchronous_fcil_reference.json --output /new/plan --data-root data
# Training only when requested:
PYTHONPATH=src python -m cafcl.reference_fcil run --config configs/synchronous_fcil_reference.json --output /new/run --data-root data --device cuda:0
```

The open-world Task-IL reference starts with no task-specific heads, registers
a head on local delivery, uses deterministic initialization, preserves SGD
momentum by parameter name, and reconciles stale schemas at actual uploads.
The shared-masked mode keeps a fixed classifier parameter count across
curricula. It evaluates integrated heads and reports coverage of arrived
heads separately. These are independent SGD/private-Replay controls, not
migrations of every legacy native algorithm or the existing joint runtime.

```bash
PYTHONPATH=src python -m cafcl.open_world --plan PLAN.json --output /new/open --mode dynamic --device cuda:0
PYTHONPATH=src python -m cafcl.open_world --plan PLAN.json --output /new/shared --mode shared_masked --device cuda:0
```

## Matched control planning

```bash
PYTHONPATH=src python -m cafcl.joint.remediation_controls --manifest MANIFEST.json --job-id PARTIAL_REPLAY_JOB --output /new/control-plan --fixed-support 50
```

The generated manifest and contrasts hold the atomic plan constant for
archive, staleness, momentum and weighting interventions. Fixed occurrence
support changes unique-data availability explicitly. The replay/no-replay
contrast gates estimates on measured optimizer-step and fresh-draw matching;
total forward work differs if replay draws differ. Do not describe this as
equal FLOPs. Confirmation slots remain reserved until specified.

## Remaining validation and limitations

No full training campaign has been launched by this remediation pass.
The legacy workflow now reaches its regression tests rather than failing at
the DER registry validation. Its earlier failing tests include obsolete
pre-Task-IL fixtures and genuine incompatibilities. Do not suppress or
delete these tests to create an artificial green CI result. Review the
exact latest Actions runs. Dynamic heads are a separate reference engine;
legacy/native method integration, complete cross-engine protocol unification,
full semantic verification of report graphics, and exact sub-stage resume
remain open. The fixed-capacity shared-mask control changes classifier
parameter sharing, so it is not identical to independent heads.

The original historical runs and frozen ledgers are not rewritten. Use
fresh output roots and source-compatible checkpoints.
