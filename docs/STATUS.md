# Implementation status — 2026-11-18, server validation pass

**Executable and test-validated offline research prototype; NOT exhaustive research closure.**

## Verified counts

- **152** named research blocks in the master catalogue.
- **144** block IDs have at least one dedicated executable job on BOTH dataset suites.
- **8** block IDs still have no executable job: **A12, B06, B10, H08, K05, K08, M08, N06**.
- **144** block IDs have some implementation coverage: **22 marked implemented, 122 marked partial**. Partial means a real runnable mechanism or diagnostic exists but precise causal controls, fidelity, or full variants are incomplete.
- **423 CIFAR-10 configurations and 425 CIFAR-100 configurations**, all configured at batch size 512, with independent IDs and artifacts. CIFAR-100's suite includes jobs using **`cifar100_20` (10+10)** and the full 100-class dataset.
- **66 tests passed, 3 CUDA-specific tests skipped** in the latest full run on MtechDEServer1 with CUDA hidden. Coverage includes synthetic end-to-end execution, synchronization, checkpoint recovery, normalized aggregation weights, GEM/A-GEM, SI, domain/task-IL evaluation, data interventions, CUDA fail-fast, deterministic validation partitions, and bounded subset counterfactuals. The CUDA interruption/resume test is among the pending GPU checks.
- **Zero official full CIFAR training suites executed or statistically validated** during this authoring work. No paper performance numbers are claimed.
- The official CIFAR-10 archive checksum and torchvision train/test counts are verified on Mtech. A separate **71-job, batch-512, 15%-training-data GPU pilot** covers the registered mechanisms. Its sequential runner is waiting for unrelated GPU jobs to finish; no pilot job has started.

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
- Deterministic class-stratified validation partition reserved from original CIFAR training data; optional class-stratified training-only pilot subsampling; per-run source-index hashes and class counts.
- Per-round metrics, forgetting history, causal diagnostics and adaptive selection use validation data. Official CIFAR test evaluation is isolated to the final saved model.
- CIFAR planner jobs set `require_cuda=true`; no-GPU execution fails before output creation. Local SGD updates, global aggregation/server optimization, GEM dual solve and model evaluation use the selected CUDA device. Runtime device and communication costs are recorded.
- Every successful round records normalized per-client aggregation weights and validates their finiteness and sum. The pilot verifier checks these alongside round counts, CUDA execution, validation-only feedback, synchronization versions, communication accounting, checkpoints, and reloaded model inference.

## Important scientific qualifications

1. **Per-round global validation population includes every arrived client/class**, irrespective of participation. Domain-incremental runs evaluate arrived client-domain slices. The official test set is evaluated only at the final model; a never-globally-learned class reduces all-observed accuracy but is not automatically called forgotten.
2. Every active selected client gets the global weights **before training and immediately after every successful aggregation**. Private historical memory persists unless a deliberate destructive ablation requests reset.
3. Class-set metadata and diagnostic gradients used in certain selection policies are simulator-privileged. These are oracle/simulation baselines; there is no secure aggregation or differential privacy guarantee.
4. ER-ACE, DER/DER++, EWC, MAS and LwF remain adaptations, not independently validated paper reproductions. The present dynamic-arrival/task-free setting is **class-aware and boundary-free**, not unsupervised class discovery.
5. The `F20` and `L03/L04` subset engine enumerates **already computed participating-client updates only**, not all eligible clients or future trajectories. Full multi-round causal forks exist for targeted include/exclude interventions, but not all possible causal comparisons.
6. Client/model/time/data allocations in all runs require empirical validation on official CIFAR and GPU hardware. Batch size 512 is a configuration invariant for CIFAR, not proof of full training success.
7. Not every job has compute-, class-exposure- and plasticity-matched counterfactual arms. Several of the new 71 block IDs are only partial research controls.
8. The latest full test suite, exact plan-manifest validator and 423-job plan-only gate have passed on MtechDEServer1. GPU GEM/resume, live-CIFAR pilot execution, hyperparameter tuning and production jobs remain unverified.

## Remaining research closure

The eight unscheduled blocks need implementations (A12, B06, B10, H08, K05, K08, M08, N06). In addition, the **122 partially covered blocks** require varying degrees of deeper implementations, exact causal controls, full reference fidelity, robust privacy/communication restrictions, task-free boundary detection, actual open-world class discovery, dynamic classifiers, higher-order causal comparisons and other planned variants.

To achieve genuine scientific closure: complete missing and partial variants; reproduce strong contemporary FCL baselines (e.g. SCAFFOLD, GLFC and appropriate peers); run the full official CIFAR experiments with multiple independent replications; verify all stage-by-stage outputs; perform negative controls and statistical comparisons; publish reproducible figures, tables, checkpoints and limitations.

For exact per-block implementation status and jobs see [`EXPERIMENT_CATALOGUE.md`](EXPERIMENT_CATALOGUE.md).
