# Essential joint campaign — 60 jobs, seed 0 only

This is the default production campaign. The larger experimental catalog
remains available through explicit --suite all, but is not scheduled here.

| Group | Designs | Task-IL | Class-IL | Jobs |
|---|---:|---:|---:|---:|
| Smoke | 4 | 4 | 4 | 8 |
| Foundational S1/S2/S3 × O0/O1 × L1/L2/L5 | 18 | 18 | 18 | 36 |
| Screening D-G5, E-G5, N-S3-random-1, N-S3-unique-1 | 4 | 4 | 4 | 8 |
| S3-O1 controls (no replay, pooled, local-only, centralized) | 4 | 4 | 4 | 8 |
| Total | 30 | 30 | 30 | 60 |

All conditions use one seed (0). Retries, if explicitly enabled, use the
same seed and resume existing jobs. Full factorial, extension and
intra-task suites are not launched. The compiler fails rather than silently
replace, drop or add essential conditions.

From the repository root, with the existing dependencies installed:

    PYTHONPATH=src python -m cafcl.joint.runner campaign --output runs/joint_essential --gpus 0 1 --min-free-mib 16000

GPU IDs and free-memory threshold are examples, not hardware measurements.
This runs planning and validation, all eight smoke jobs, then (only if all
eight pass artifact audit) the remaining 52 jobs, then audit, per-job reports
and comparative analysis. No extra seed or job is automatically introduced.
The broker schedules one worker per leased GPU, and uses the latest-only
checkpoint policy. It enables local lifecycle evaluation; expensive
mechanistic probes are opt-in. Repeat the same command with the same clean
source revision, root and execution options to resume; already verified jobs
are skipped. Use --plan-only for a no-training dry run.

Outputs under runs/joint_essential:
- plan/manifest.json and plans/: immutable job definitions and compiled plans
- plan_validation.json and campaign_identity.json
- job_ledger.json and job_ledger.csv: current state of every canonical job
- results/<job_id>/: metrics, checkpoints, reports, diagnostics, analysis
- results/broker_receipts/: persistent attempt logs and receipts
- results/joint_campaign_audit.json and .csv
- comparison/: paired results, missing arms and matching checks
- campaign_summary.json

Ledger updates on worker start, result and final audit; attempt receipts
remain separate and are not deleted on resume. The ledger is not a periodic
GPU utilization monitor. No extra scientific runs are selected based on
test results. Smoke accuracy is not a performance acceptance threshold.
