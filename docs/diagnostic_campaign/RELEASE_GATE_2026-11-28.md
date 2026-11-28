# Diagnostic campaign release gate — 2026-11-28

## Decision

**Not released for production execution.** Repository and factorial-plan validation now pass, but the ten CF confirmation slots still lack scientific bindings. Real CIFAR-100 GPU smoke, failure/recovery validation, and a durable two-host coordinator have not been completed. No diagnostic production run was admitted from this worktree.

## Source and inventory

- Source revision inspected and currently pushed on `main`: `5b19cc51df629d42d90c362f32337726f6883dee`.
- The current candidate is that source plus the uncommitted changes listed in the issue register and Git worktree.
- The canonical ledger contains 152 slots. The local and remote real-data preflight compiled 142 configured jobs and reported 10 reserved CF slots. This is configuration preflight, not GPU execution or result evidence.
- No CF config, seed, selection rule, or binding is defined for the 10 reserved slots. Keep them explicitly specification-blocked until the author supplies definitions.
- The original 53-issue audit descriptions are absent from this checkout. The register preserves 53 issue identities and summary dispositions, but explicitly does not claim independent issue-by-issue resolution.

## Validation on the current candidate worktree

- `.venv/bin/pytest -q --tb=no`: **458 passed, 335 subtests passed, 8 warnings** (238.96 s).
- `.venv/bin/pytest -q tests/test_joint_*.py --tb=no`: **70 passed, 335 subtests passed** (62.68 s).
- `.venv/bin/python scripts/validate_factorial_plans.py`: passed. It compiled 32,133 valid jobs per model, 99 method/aggregator pairs, 53,805 inputs, and 21,672 rejections for each ViT/CNN plan; current primary-probe expectations are five concurrent clients and 20 intervals.
- `.venv/bin/python scripts/audit_stream_math.py`: passed with the same 53,805 / 32,133 / 21,672 input, accepted, and rejected totals.
- The full run emitted eight warnings, including existing upstream source syntax/deprecation warnings and one test tensor-to-float warning. No test failed.
- CI for the earlier code push at `3a26705` had a passing joint workflow and a failing repository-wide test workflow. [Run #204](https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/46151262520). That CI result predates this candidate’s corrections; current local success is not yet a new remote CI result.

## Corrections verified in this candidate

- Non-stream arrival events now provide `class_ids` to downstream event preparation.
- Task-IL EWC/MAS importance consolidation uses active task-head output coordinates.
- Task-IL GEM/A-GEM reference gradients retain the physical forward batch while excluding padding from the loss.
- Spurious reversed-test evaluation now receives task identities, so local head columns are not treated as global class IDs.
- Synthetic Task-IL label noise now selects replacement labels from each sample’s source-task label set.
- Invalid singleton Task-IL test fixtures were rebuilt with valid multi-class schedules; historical direct Class-IL API checks are explicitly marked or isolated. Native GLFC/LGA/FEAT Task-IL pilot construction now remains blocked because those methods lack a task-head adaptation.
- Factorial snapshot fingerprints/examples and stale primary-probe expectations were refreshed without changing validated plan counts or scientific configuration values.

## Remaining release requirements

1. Locate the original 53-issue audit descriptions and independently map each finding to source, tests, and ledger jobs.
2. Obtain scientific definitions for the ten CF slots or retain them as specification-blocked.
3. Complete deterministic synthetic integration and failure/recovery checks across the production engines and coordinator.
4. Perform a minimal real CIFAR-100 GPU smoke on each authorized server, including independent artifact validation.
5. Establish and verify one persistent two-host coordinator with atomic claims, attempt identities, resource reservations, recovery, and artifact checksums.
6. Commit and push the verified candidate, obtain fresh CI evidence, and pin both servers to that exact revision before production admission.
