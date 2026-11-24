# Unified 152-slot FCL diagnostic campaign

Canonical [JSON ledger](../../configs/diagnostic_campaign/job_ledger.json) and [CSV mirror](../../configs/diagnostic_campaign/job_ledger.csv). Separate from the historical 121-job campaign and joint essential suite.

## Scope and execution fidelity

| Family | Jobs | Binding |
|---|---:|---|
| FL | 11 | Stationary reference |
| MC / CM / MO | 60 | Matched reference v1 |
| CC / JR / OF / ST / JC | 71 | Explicit historical reconstruction v1 |
| CF | 10 | Reserved, not executable until configured |

142 fixed configurations have a reference code path; ten confirmation slots are reserved. The 71 historical-dependent conditions, including 14 intended reproductions, are **not exact historical reproductions**: original V25/V27 runtime source variants are absent from main. They retain their original planned IDs but run only with explicit `--allow-reconstruction`; exact historical-fidelity mode remains blocked. Do not compare a reconstructed intervention with an archived baseline as though the only difference is the intervention. Compare against the reconstructed baseline from this suite. The result metadata records the fidelity boundary.

## One-command execution

From a clean checkout with the project dependencies installed:

```bash
git pull origin main
PYTHONPATH=src python -m cafcl.diagnostic_campaign plan --all --allow-reconstruction
PYTHONPATH=src python -m cafcl.diagnostic_campaign preflight --all --allow-reconstruction --data-root data
PYTHONPATH=src python -m cafcl.diagnostic_campaign run --all --allow-reconstruction --gpus 0 1 --data-root data --retries 1
```

Use actual visible GPU indices; for one GPU use `--device cuda:0`. The suite uses one worker process per GPU and assigns the next pending job to the first free device. There is no CPU fallback. `--download` is optional and fetches the data once during preflight. `--subset-probes` enables computationally expensive all nonempty cohort diagnostics for K<=5 and changes the effective configuration signature.

Resume at the job level using the same flags and output root:
```bash
PYTHONPATH=src python -m cafcl.diagnostic_campaign resume --all --allow-reconstruction --gpus 0 1 --data-root data --retries 1
PYTHONPATH=src python -m cafcl.diagnostic_campaign status --all --allow-reconstruction
```

Without reconstruction opt-in, select `--family FL`, `MC`, `CM`, or `MO`, or use `--ready-only` to execute 71 independent reference jobs. `--wave 1`, `--family JR`, and `--job-id CC-001` filter the same canonical ledger. Reserved slots are skipped. The suite is sequential within each GPU job; client models are evaluated/trained sequentially using saved state, not simultaneous CUDA streams. The scheduler does not estimate VRAM or support multiple hosts sharing the same suite lock.

## Results

`runs/diagnostic_campaign/<family>/<job-id>/attempt_NNNN/` contains immutable job spec, commit/device provenance, status, logs, partition and schedule, metrics, diagnostics, checkpoints and terminal summary. The suite validates expected round count, sequential IDs, terminal summary identity, output files and final accuracy before completion. It writes central `suite_status.json`, `suite_status.csv`, `last_launch.json` and per-attempt `artifact_validation.json`. Job retries restart from scratch; this is not full-state mid-job continuation. An interrupted active lock must be cleared only after verifying its worker has stopped. Use a new output root for a new scientific source revision; configuration signatures separate reconstruction and diagnostic options.

## Protocol notes

- FL: stationary 30-class fixed subset, same data across K, common or restricted CE, sample-weighted FedAvg, reset SGD momentum each block, deterministic preprocessing. Central reference matches single-client's 100 blocks ×10 epochs. Validation each block, official test at checkpoints; no test feedback.
- MC/CM: 100 classes, ten classes per stage, ten stages, every unique TRAIN index once. IID or 80% dominant class mixture +20% uniform, with allocation audits. Global observed-class CE or semantic binary heads. For CM the parent-stage replay archive is frozen and epoch-indexed batch stream is unchanged across R=1/5/10; logs actual steps/current/replay draws.
- MO: disjoint and complete-task overlap share semantic pairs; partial overlap changes the second matching. Two 450-sample units per client per stage, same seeded cohorts at fixed participation, fixed 50/class shared replay cap. Partial overlap changes TIL discrimination objectives.
- Historical reconstruction: archived class sequences and exact archived 40%/60% cohorts (or explicit OF cycles); unique occurrence-specific heads, final duplicate pair in joint chain, local-known competitor CE in original CIL, global observed CE intervention. Repeated rounds preserve parent task/cohort. Replay cap 50/class occurrence; deterministic independent sample partition and crop/flip; paired physical-batch padding. Task-ID accuracy and documented class-membership union evaluation are distinct outputs.
- ST: beta 0/2; fresh-only discards stale pending upload and refreshes selected clients while retaining private replay; pre-train download discards unsent displacement; reset momentum; TRAIN-only BN recalibration on received data/shared replay. These are explicitly new diagnostic policies, not verified original mechanisms.
- FCL: persistent client SGD unless reset intervention, sample-weighted full-model FedAvg, stale parameter rebasing, stale BN buffers retained. Shared replay is simulated TRAIN-index sharing and is not privacy preserving. Global-seen objective shares current label metadata even with offline clients; no future labels enter the loss.
- Diagnostics: local/pre/post aggregation/download fixed validation panels, sampled current and current-plus-replay gradient norm and backbone cosine before and after each communication, historical gradient-to-update alignment, candidate-update geometry; official test by stage and terminal flush; optional subset evaluation. Full exact population gradient norm, full-state checkpoint/resume, and automatic inferential multi-seed confirmation remain open. No GPU validation is claimed by the authoring session.


## Central scientific analysis

The run suite automatically generates `analysis/job_summary.csv`,
`learning_trajectories.csv`, `controlled_contrasts.csv`,
`compute_matching.json` and `analysis_summary.json` after execution.
Rebuild without training:
```bash
PYTHONPATH=src python -m cafcl.diagnostic_analysis --allow-reconstruction
```
Use the same `--subset-probes` flag if used for execution. The report reads
only terminal artifact-validated attempts with matching effective job specs.
Acquisition-conditioned retention uses 50/70/90% thresholds and a fixed
mid-horizon cohort. CM reports actual budget equality, not nominal epochs.
All contrasts are descriptive at seed zero; no test results drive training.
The optional subset probes evaluate all nonempty cohorts (K<=5) and
parameter-only pair nonadditivity for actual uploaders using fixed base BN
buffers. Gradient norms use fixed sampled TRAIN panels, not full exact
population gradients. Historical directional probes use fixed validation
panels. Exact full-population-gradient convergence claims remain future work.


## Fixed-checkpoint matched acquisition diagnostic

The two MC K=2 IID R=1 arms save a common historical reference after stage
five. After those anchors are artifact-validated, the optional
`--fork-probes` flag runs 12 short next-stage branches: 2 IL protocols ×
2 partitions × K={2,3,10}. Each branch begins with the identical historical
checkpoint and replay archive for its IL protocol, reset local optimizer
states, identical next-stage data, ten communications × two local epochs.
These are additional diagnostic branches, not extra full-training ledger
slots. Results include fixed historical accuracy loss at the earliest
measured 50/60/70/80/90% validation acquisition crossing; unreachable
thresholds are reported as such. The final official test is never used to
select thresholds or train.

```bash
PYTHONPATH=src python -m cafcl.diagnostic_campaign run --all --allow-reconstruction --subset-probes --fork-probes --gpus 0 1 --retries 1
```

Or independently after the anchors exist:
```bash
PYTHONPATH=src python -m cafcl.diagnostic_forks --allow-reconstruction --subset-probes --device cuda:0
```

A result root is pinned to its tracked Git revision and ledger digest;
use a different output root when changing code. Terminal artifact receipts
are checked against the recorded checksums when a completed job is reused.
