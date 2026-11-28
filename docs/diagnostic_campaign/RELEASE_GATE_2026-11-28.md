# Diagnostic campaign release gate — 2026-11-28

## Decision

**Not released for production execution.** The repository-wide test gate is red and the ten CF confirmation slots still lack scientific bindings. No diagnostic production run was admitted in this worktree.

## Source and worktree

- Base source revision inspected: `30b5f593b77d802cfe25a2b504e731f802f332c7` (`main`).
- Candidate revision: pending commit of the focused corrections recorded in this worktree.
- The 152-slot canonical ledger remains the inventory. Local and remote real-data preflight previously compiled 142 configured jobs and reported 10 reserved slots; this is configuration preflight, not GPU execution or result evidence.
- The ten CF slots have no config, seed, selection rule, or binding in the checked-in ledger. They remain specification-blocked pending the author's definition.

## Validation evidence on the candidate worktree

- `pytest -q tests/test_joint_*.py`: **70 passed, 335 subtests passed** (43.72 s).
- Focused class-indexed simulator suite, including strict singleton rejection and all 11 method parameters: **12 passed** (4.37 s).
- EWC/MAS importance-coordinate focused tests: **2 passed** (2.59 s).
- Full `.venv/bin/pytest -q`: **108 failed, 349 passed, 335 subtests passed, 8 warnings** (62.98 s). The repository-wide gate therefore fails.
- `scripts/audit_stream_math.py` passed on the earlier candidate inspection; `scripts/validate_factorial_plans.py` failed because the generated v16 registry hash does not match the frozen snapshot. Neither result is a production release clearance.
- Current-job CI on base `30b5f59` also had a failed main tests workflow and a successful isolated joint workflow (see run links in the campaign audit notes).

## Confirmed fixes in this candidate

- Non-stream arrival events now provide the `class_ids` field consumed by event preparation; targeted cross-method tests pass.
- Task-IL EWC/MAS consolidation now indexes importance estimates in active task-head coordinates; targeted regressions pass.
- The offline-client replay assertion now matches the declared selected-client archive policy and verifies that remote archive content is not attributed to an offline recipient.
- Strict Task-IL singleton rejection retains its scientific guard and now reports the discriminative-class requirement clearly.

These targeted passes do not establish repository-wide correctness. The issue register retains the documented limitations and does not treat all 53 summarized issues as independently resolved.

## Required before production

1. Investigate and resolve or rigorously isolate every repository-wide test failure that can affect the supported campaign pathways; rerun the complete suite and required validators.
2. Reconcile the v16 factorial manifest hash mismatch without weakening frozen-plan validation.
3. Obtain/locate the original 53-issue audit descriptions and independently map the findings to source, tests, and ledger jobs.
4. Define the ten CF confirmation slots or keep them explicitly blocked; do not count them as verified runs.
5. Complete synthetic integration, real CIFAR-100 GPU smoke, and failure/recovery validation before accepting a campaign revision.
6. Establish and verify a persistent, transactional two-host coordinator with atomic claims, attempt identities, resource reservations, and artifact verification.
7. Commit and push the validated revision, then pin both servers to it before production admission.
