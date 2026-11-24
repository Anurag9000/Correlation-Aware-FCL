# Audit remediation (November 2026)

This record tracks the 53 concerns from the static audit of c247e395. The original
audit mixed actual defects, experimental assumptions, and requests for new
controls. It is not a list of 53 independently reproduced bugs.

Protocol identifiers were deliberately advanced. Do not resume old checkpoints
or pool old and new numerical results. The original archived runs are unchanged.
New source/data/dependency receipts apply to new simulator runs. Historical
reconstructions are still reconstructions, not exact upstream reproductions.

## Issue register

Status: **implemented** means a code path or correction was added; it does not
mean the relevant GPU experiment has run. **partial** means a control or
measurement exists but complete scientific validation remains open.

| ID | Disposition |
|---:|---|
| 1 | Implemented: independent task-conditioned evaluation, separate legacy oracle. |
| 2 | Partial: comparisons prefer standard Task-IL; differing task widths/geometry still define different objectives. |
| 3 | Implemented: explicit task-head gradient objectives and contributor-head counterfactual aggregation; verify optional legacy paths. |
| 4 | Partial: independent archive, staleness, momentum and weighting switches; matched factorial experiments still required. |
| 5 | Implemented: joint model-only federation with private replay; privileged simulator metadata is not a privacy guarantee. |
| 6 | Implemented: actual candidate-universe chance and fixed-head validation peak drops; legacy class peak drops remain descriptive. |
| 7 | Implemented: fixed unique-data and fixed per-occurrence support are distinct compiler contracts. |
| 8 | Implemented: parameter-only gradient/update geometry in simulator aggregation. |
| 9 | Implemented: damped/undamped/direct stale updates and terminal guard control. |
| 10 | Implemented: joint variable batches; legacy fixed-padding protocols retained. |
| 11 | Partial: accumulated replay uses actual valid microbatch weighting; compute-matched contrasts still need execution. |
| 12 | Implemented: reset on synchronization or parent task as distinct joint controls. |
| 13 | Implemented: explicit audit_test_every_sync=False and primary strict terminal-only test. Historical audit policy remains opt-in/default-compatible for old configurations. |
| 14 | Partial: matched diagnostic reference uses synchronized ten-class stages; a fully independent Dirichlet-selected synchronous FCIL benchmark is still needed. |
| 15 | Implemented: primary seed parameter; five-seed confirmation experiments not executed. |
| 16 | Partial: protocol versioning and preserved historical roots; frozen legacy registry remains incompatible with Task-IL DER raw reuse. |
| 17 | Existing newer main: 142 reference bindings, 10 reserved; 71 historical reconstruction jobs require opt-in. |
| 18 | Implemented: DER logit width derived from task head class map. |
| 19 | Implemented: optional deterministic bounded server reservoir, distinct from historical append-only archive. |
| 20 | Implemented: parent experience IDs and delayed private memory consolidation in batch sync. Exact draw matching remains a separate experiment. |
| 21 | Implemented: privileged arrived-data selector with seeded no-evidence fallback. |
| 22 | Implemented: delivered/optimized/received ledgers and holdings selector; legacy knowledge-scope fields retained for compatibility. |
| 23 | Implemented: fresh-example or optimizer-step FedAvg weights; no claim that either is universally optimal. |
| 24 | Implemented: distinct-first versus class-uniform-with-replacement replay. |
| 25 | Partial: explicit head policy and count in plan; equal-capacity architecture controls remain to be designed. |
| 26 | Open: preallocated future heads are a closed-world assumption; dynamic expansion across stale clients is not implemented. |
| 27 | Implemented: compiled learning horizon in participation scheduler. |
| 28 | Implemented: actual microbatch count/valid-size accumulation; masked legacy partial replay restricted to ratio 1. |
| 29 | Implemented: actual delivered class ledger distinct from declared task labels. |
| 30 | Implemented: acknowledged optimized-index ledger, separate from delivered indices; replay exposure remains separately counted. |
| 31 | Partial: matched reference exposes local/global competitor controls; calibration experiments remain needed. |
| 32 | Implemented: explicit pre-terminal-flush metrics; strict primary test only after flush, pre-flush validation available. |
| 33 | Implemented: binary arrival presence rather than counts. |
| 34 | Partial: random seeded gradient pairs/panels; legacy feature panels still take first eligible indices. |
| 35 | Implemented: joint batch size configurable; upstream native/legacy fixed batch contracts retained. |
| 36 | Implemented: refuse optimized Python (-O) that removes validation assertions; assertion-to-exception migration remains optional cleanup. |
| 37 | Implemented: server score tables labelled as global, joint private-model evaluation separately available. |
| 38 | Implemented: full server-state tensor/payload hash. |
| 39 | Existing newer main: completed attempts validated; missing artifacts reported invalid. |
| 40 | Implemented: explicit stationary policy validation. |
| 41 | Implemented: reconstruct and compare complete joint execution config and launch receipt. |
| 42 | Implemented: source, dependencies, raw data and curriculum checkpoint identity. |
| 43 | Implemented: raw pixel/tensor and split mapping fingerprint. |
| 44 | Implemented: raster report limited to latest 30 columns; full tabular history retained. |
| 45 | Implemented: model and new archive payload accounting; excludes protocol metadata/serialization, not real network measurement. |
| 46 | Implemented: atomic-step and communication-interval latency separately. |
| 47 | Implemented: occurrence-weighted and interval-weighted metrics separately. |
| 48 | Implemented: vertical-flip domain intervention avoids cancellation by horizontal augmentation; rerun domain experiments. |
| 49 | Implemented: event-causal head exposure/received provenance for Task-IL. |
| 50 | Implemented: exact head/class/training-interval and download anchor receipt checks. |
| 51 | Implemented: training anchor versions captured before synchronization. |
| 52 | Partial: stationary resume from per-round durable checkpoints; matched/reference job retry is still job-level. |
| 53 | Partial: independent final checkpoint inference verifier added for supported canonical runs; CSV/PNG/MD content verification remains limited. |

## New independent joint controls

Manifest jobs can specify:
`share_continual_state=false` (private replay/model only),
`server_memory_per_class=50` (positive shared cap),
`stale_update_policy=damped_delta|undamped_delta|direct_model`,
`terminal_stale_guard=false`,
`optimizer_state_policy=persist|reset|reset_on_sync|reset_on_task`,
`aggregation_weight_policy=fresh_examples|optimizer_steps`,
`physical_batch_policy=variable`, `batch_size=32|64|128`,
`replay_update_policy=sampled_mixed|sampled_mixed_accumulated`, and
`replay_class_sampling_policy=distinct_first|class_uniform_with_replacement`.
These are distinct interventions. Copy a manifest job with a new unique job ID,
change one field, and preserve its plan hash for a matched contrast. Do not
combine all changes into a single causal comparison.

The compiler supports `samples_per_class_occurrence=N` as a separate
fixed-support allocation contract, with unused data explicitly recorded.
It rejects infeasible budgets rather than duplicating samples silently.

Primary paired replication (run one pair per seed, distinct output roots):
```bash
for seed in 0 1 2 3 4; do
  PYTHONPATH=src python -m cafcl.primary_probe --seed "$seed" --gpu 0
done
```

Independent final inference, on a trusted checkpoint with matching source,
dependency and dataset identity:
```bash
PYTHONPATH=src python -m cafcl.verify_final --run-dir PATH --data-root data --device cpu
```

Stationary campaign resume now uses compatible per-round checkpoints when
available; it rejects a mismatched job/source/data identity. Nonstationary
matched/reconstruction workers still restart.

## Validation boundary

The audit commits are source changes, not completed experimental evidence.
The joint CI workflow and the real-data smoke workflow must be checked for the
exact commit. Legacy frozen-ledger CI was failing before these changes because
it includes ambiguous Task-IL DER/DER++ raw-reuse settings; do not bypass the
validator to make that ledger look green. No full GPU campaign, dynamic-head
migration, strict private asynchronous production deployment, or five-seed
statistical validation is claimed.
