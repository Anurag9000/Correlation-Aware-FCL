# Unified diagnostic campaign (152 slots)

This is a **separate** campaign for client-count scaling and the joint/disjoint reversal. It does not replace the archived 121-job campaign or the existing joint suite.

Canonical ledger: [job_ledger.json](../../configs/diagnostic_campaign/job_ledger.json). CSV is the inspection mirror. Counts: 142 configured jobs (14 reproductions), 10 reserved confirmation slots.

## Run suite
From the repository root, after installing the repository dependencies:
```bash
PYTHONPATH=src python -m cafcl.diagnostic_campaign plan --all
PYTHONPATH=src python -m cafcl.diagnostic_campaign preflight --family FL --data-root data
PYTHONPATH=src python -m cafcl.diagnostic_campaign run --family FL --device cuda:0 --data-root data
PYTHONPATH=src python -m cafcl.diagnostic_campaign resume --family FL --device cuda:0 --data-root data
PYTHONPATH=src python -m cafcl.diagnostic_campaign status --all
```

`run --all` **fails closed before any training** while any selected slot lacks a binding. `run --all --ready-only` explicitly executes only implemented bindings. This avoids silently running a different scientific condition. The current implementation binds the 11 stationary FL reference jobs; 131 configured FCL jobs are blocked, and 10 confirmation slots are reserved. No GPU run is claimed verified. The FL reference uses the repository data split and ResNet-18 global classifier, a fixed 30-class subset, sample-weighted averaging and reset SGD momentum each block; centralized reference has 100 blocks of 10 epochs, matching the single-client FL trajectory budget. Shared deterministic preprocessing (no augmentation) is deliberate for this independent reference. Roundwise validation and offline test checkpoints are logged; test metrics never drive decisions. Exact gradient-norm and compute-matching diagnostics are not implemented here.

Results default to `runs/diagnostic_campaign/<family>/<job-id>/attempt_NNNN/`. Each attempt has job and source provenance, logs, status, partition, metrics, checkpoints and summary. Completed attempts are skipped, failed ones can be rerun in a fresh attempt; this is job-level retry, not mid-training resume. The runner is sequential, not a multi-GPU broker. An interrupted active lock requires manual verification and removal.

## Remaining runtime bindings

- CC/JR: restore exact historical V25/V27 runtime and schedules from provenanced sources, not a current joint substitute.
- CC/MC/CM/JC: implement the shared globally observed class objective and repeated same-stage communication under the stated budgets; match actual sample and optimizer exposures.
- OF/ST: exact uploader schedules and isolated stale-update interventions, including diagnostics.
- MO: matched ownership with pure disjoint support; current joint compiler rejects no cross-client overlap.
- CF: freeze confirmation configurations after evidence.
- Implement fixed-cohort current/history evaluation, local/aggregation/download forgetting, per-head metrics, gradient norm/alignment, checkpoint forks, counterfactual subset evaluation and validation gates before calling the complete campaign runnable.

The existing joint epoch-slicing path is restricted to full participation and divides a fixed epoch budget; it cannot be silently substituted for the proposed partial-participation repeated-optimization jobs. No existing training files or historical ledgers are changed.
