# CIFAR-100 continuous local-task status (v3)

## Implemented and offline verified

- A single canonical class-indexed 100-output classifier; no task-head creation or inference task IDs in the new CIFAR-100 registry.
- Local episode cross-entropy restricted to currently active class IDs; other classifier rows frozen (including against optimizer momentum).
- Configurable `classes_per_local_task` and `local_tasks_per_round`; default two consecutive binary tasks before each server update.
- Continuous learning and durable private CL state for SGD, GEM, A-GEM, replay, EWC, SI, ER-ACE, DER, DER++, MAS, LwF, even when not selected to communicate.
- Class-label overlaps within and across clients, in the same and different intervals, and combined; distinct raw training samples by default and explicit raw overlap where requested.
- Exactly count-weighted canonical class-row fusion at the server, accumulated offline stale-delta upload, post-aggregation active-client replacement, and no inactive-client sync.
- Independent per-round random participation, exact-D controlled offline cohorts, and an optional communication-only final integration step.
- CIFAR-100-only experiment registry covering 53,805 distinct configurations and all 99 local/server method pairs; old CIFAR-10 and 20-class suites retired from active launch paths.
- Synthetic integration tests and pure 100-class scenario tests validate the user's specific class-to-client schedule, all overlap types, no unrequested raw-sample reuse, offline recovery, and method-specific local history.

## Not established

- **No full official CIFAR-100 GPU sweep** has been run, and no official test accuracy or research comparison is claimed.
- The registered suite is exhaustive only for its declared *finite* experiment axes, not every possible combination of independent continuous hyperparameters.
- The global model has allocated but inactive class rows from initialization rather than dynamically growing tensor shapes; class-wise merging is an experimental weighting heuristic and must be empirically calibrated.
- Freezing nonactive classifier rows does not freeze hidden/backbone representation learning, so interference and forgetting remain possible.
- Regularization, GEM projection, replay/distillation variants, and class-specific server weighting are not independently verified as exact paper reimplementations; GEM's CPU SciPy dual solver is still used even if neural-network training executes on CUDA.
- Some selection policies are privileged simulation/oracle controls. Training on official CIFAR validation/test feedback must be separated before publication claims.
- The old 152-block archival method catalogue is not proof that every previously imagined research block exists in the new stream. Removed old job manifests were scientifically incompatible with the current task-stream requirements.

## Regeneration

Run `python scripts/validate_factorial_plans.py` to validate the source-generated plan and the design's SHA256. The production runner writes `results/cifar100_stream_v3/stream_manifest.json`, independent per-job artifacts, durable full-state checkpoints, and protocol-specific success markers. Never count checkpoints from earlier protocols as completed new jobs.
