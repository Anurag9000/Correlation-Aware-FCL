> **2026-11-19 protocol correction:** [Continuous local CL for all methods](CONTINUOUS_LOCAL_CL_PROTOCOL.md) replaces selected-client-only training. Unselected clients learn privately; the server receives their accumulated model updates only upon reconnection. Old run/checkpoint protocol versions are rejected. The 67,938/69,722 configured jobs remain unexecuted.\n\n> **2026-11-18 expansion:** The default launchers now register **67,938 CIFAR-10** and **69,722 CIFAR-100** finite-grid jobs, including the previous 423/425 targeted experiments; all 99 local-method/server-method pairings are covered at batch size 512. None of the new sweeps has been trained. Methodology limitations below remain. See [FACTORIAL_SUITE.md](FACTORIAL_SUITE.md).
# Implementation status — 2026-11-17, expanded engineering pass

**Executable and test-validated offline research prototype; NOT exhaustive research closure.**

## Verified counts

- **152** named research blocks in the master catalogue.
- **144** block IDs have at least one dedicated executable job on BOTH dataset suites.
- **8** block IDs still have no executable job: **A12, B06, B10, H08, K05, K08, M08, N06**.
- **144** block IDs have some implementation coverage: **22 marked implemented, 122 marked partial**. Partial means a real runnable mechanism or diagnostic exists but precise causal controls, fidelity, or full variants are incomplete.
- **423 CIFAR-10 configurations and 425 CIFAR-100 configurations**, all configured at batch size 512, with independent IDs and artifacts. CIFAR-100's suite includes jobs using **`cifar100_20` (10+10)** and the full 100-class dataset.
- **79 offline tests passed** after the 2026-11-19 shared continuous-local-CL correction, including 15 dedicated delayed-reconnection and all-11-local-method regression tests. This does not establish real CIFAR training performance.
- **Zero official full CIFAR training suites executed or statistically validated** during this authoring work. No paper performance numbers are claimed.

## Newly wired functionality

- Default **20 communication rounds per experience** (with explicit shorter diagnostic runs where configured).
- Explicit client/class assignments and CIFAR-100 **20-class, 10+10** mapping with class 0 related to 12 and class 1 related to 13 by official coarse-superclass membership; CIFAR-10 analogues use its documented heuristic automobile/truck and cat/dog pairings.
- Per-class training dose, client sample-count skew, raw-sample overlap, class imbalance, round-level temporal recurrence and periodic recurrence, exact per-round participation overrides, unequal client arrival lags.
- Stationary and domain/gradual-drift streams, client-stage image shifts, sample-level transformed training and full client/domain held-out evaluation, task-ID-aware training/evaluation, class-aware boundary-free local memory constraints, deterministic training-label noise and spurious image markers.
- Local Synaptic Intelligence with stage consolidation, GEM class/stage/all constraint grouping, momentum reset/persistence and optional deliberate memory reset on synchronization.
- Additional privileged train-only feature/gradient relationship selectors, forgetting/history/GEM-informed client scheduling, feedback-conditioned participation.
- Class-balanced, inverse observed-participation, conflict-aware weighting, historical-gradient retention weighting and pairwise client-update projection aggregators. All are **experimental approximations**, not secure or privately deployable algorithms.
- Bounded retrospective aggregation subset enumeration, pair interactions, class-gradient diagnostics, optional pre/post historical losses, old-feature drift, communication bytes, per-round runtime, and full observed global accuracy/forgetting separation.
- Tests and CI with a validator requiring both committed batch-512 job manifests to match the actual source planner.

## Important scientific qualifications

1. **Global test population includes every arrived client/class**, irrespective of participation. Domain-incremental runs evaluate arrived client-domain slices. A never-globally-learned class reduces all-observed accuracy but is not automatically called forgotten.
2. **All data-bearing clients train locally in every round, including hidden clients.** Client selection controls only communication. No normal partial-participation client downloads the new global weights until it uploads and its contribution is acknowledged; private continual-learning histories survive. The explicit `full_sync` ablation is an exception.
3. Class-set metadata and diagnostic gradients used in certain selection policies are simulator-privileged. These are oracle/simulation baselines; there is no secure aggregation or differential privacy guarantee.
4. ER-ACE, DER/DER++, EWC, MAS and LwF remain adaptations, not independently validated paper reproductions. The present dynamic-arrival/task-free setting is **class-aware and boundary-free**, not unsupervised class discovery.
5. The `F20` and `L03/L04` subset engine enumerates **already computed participating-client updates only**, not all eligible clients or future trajectories. Full multi-round causal forks exist for targeted include/exclude interventions, but not all possible causal comparisons.
6. Client/model/time/data allocations in all runs require empirical validation on official CIFAR and GPU hardware. Batch size 512 is a configuration invariant for CIFAR, not proof of full training success.
7. Not every job has compute-, class-exposure- and plasticity-matched counterfactual arms. Several of the new 71 block IDs are only partial research controls.

## Remaining research closure

The eight unscheduled blocks need implementations (A12, B06, B10, H08, K05, K08, M08, N06). In addition, the **122 partially covered blocks** require varying degrees of deeper implementations, exact causal controls, full reference fidelity, robust privacy/communication restrictions, task-free boundary detection, actual open-world class discovery, dynamic classifiers, higher-order causal comparisons and other planned variants.

To achieve genuine scientific closure: complete missing and partial variants; reproduce strong contemporary FCL baselines (e.g. SCAFFOLD, GLFC and appropriate peers); run the full official CIFAR experiments with multiple independent replications; verify all stage-by-stage outputs; perform negative controls and statistical comparisons; publish reproducible figures, tables, checkpoints and limitations.

For exact per-block implementation status and jobs see [`EXPERIMENT_CATALOGUE.md`](EXPERIMENT_CATALOGUE.md).
