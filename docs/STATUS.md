# Implementation status — 2026-11-18, server validation pass

**Executable and test-validated offline research prototype; NOT exhaustive research closure.**

## Verified counts

- **152** named research blocks in the master catalogue.
- **144** block IDs have at least one dedicated executable job on BOTH dataset suites.
- **8** block IDs still have no executable job: **A12, B06, B10, H08, K05, K08, M08, N06**.
- **144** block IDs have some implementation coverage: **22 marked implemented, 122 marked partial**. Partial means a real runnable mechanism or diagnostic exists but precise causal controls, fidelity, or full variants are incomplete.
- **423 CIFAR-10 configurations and 425 CIFAR-100 configurations**, all configured at batch size 512, with independent IDs and artifacts. CIFAR-100's suite includes jobs using **`cifar100_20` (10+10)** and the full 100-class dataset.
- **70 tests passed, 3 CUDA-specific tests skipped** in the latest full run on MtechDEServer1 with CUDA hidden (14.02 s). This includes the new named-job selection and host-memory gate checks. The three CUDA-only checks were separately run on the V100 and passed: device-resident GEM QP, CUDA aggregation/server update, and interrupted/resumed CUDA equivalence.
- **Zero official full CIFAR production suites executed or statistically validated** during this authoring work. No paper performance numbers are claimed.
- The official CIFAR-10 archive checksum and torchvision train/test counts are verified on Mtech. A separate **71-job, batch-512, 15%-training-data GPU pilot** completed sequentially with zero failures. An independent audit verified every pilot configuration, round, validation-only feedback record, CUDA device record, normalized aggregation weight, sync version, communication total, checkpoint dependency, finite tensor, and reloaded CUDA model: **71/71 passed**. These short runs establish execution integrity only, not tuned performance or full-schedule learning.
- The eight-candidate short, validation-only SGD screen is complete and independently verified **8/8**: four learning rates (`0.003, 0.01, 0.03, 0.1`) × two momenta (`0, 0.9`), 50% of eligible training data, eight rounds per experience, batch 512. The top short-run candidate was `lr=0.01, momentum=0.9` (mean old/new validation accuracy 21.1%/15.3%; harmonic mean 17.7%). This is a finalist signal, not a production freeze.
- A six-candidate full-data plan is staged at `results/cifar10_full_validation_v2/job_plan.json`: learning rates `0.01, 0.03, 0.1` × momenta `0, 0.9`, 100% of eligible data, 20 rounds per experience, batch 512. The `0.03/0` registered baseline is included. Its persistent runner is waiting without a CUDA context for an active GPU job from another Mtech project; no v2 candidate has started.
- An earlier full-data attempt has two completed outputs. `lr=0.01, momentum=0` completed before the other GPU job started (final four-round validation means: old 19.8%, new 17.4%). `lr=0.1, momentum=0` overlapped that external CUDA job and is explicitly marked potentially contaminated in `results/cifar10_full_validation_v1/execution_notes.json`; exclude it from selection and runtime comparison. The interrupted runner stopped between jobs and preserved both outputs. The runner now frees per-job allocator memory and delays CUDA initialization until resource gates pass.
- Seventeen representative production job IDs are listed in `docs/CIFAR10_GPU_VALIDATION.md`. They passed CLI selection/plan-only validation and cover the requested mechanism families; none have started yet. Their original IDs will allow reuse under `results/cifar10/runs/` by the later 423-job production suite.
- `scripts/analyze_cifar10_production.py` produced a 71-job pilot preview under `results/cifar10_gpu_pilot_v1/analysis_preview/`: 71 registry rows, 284 per-round rows, 710 per-class rows, 1,171 aggregation rows, eight paired causal trajectory rows, 60 causal per-class deltas, seven GEM summaries, 268 gradient-pair records, and per-job SVG curves/heatmaps. Sample SVGs parsed as valid XML. The pilot predates the new confusion-matrix field, so its confusion export is empty; production runs will record that field.

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
- Per-round metrics, forgetting history, full-label confusion counts, causal diagnostics and adaptive selection use validation data. Official CIFAR test evaluation is isolated to the final saved model.
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
8. The latest full test suite (70 passed, 3 skipped), exact plan-manifest validator and 423-job plan-only gate passed on MtechDEServer1. CUDA GEM/resume and the 71-job CIFAR pilot passed. Full-schedule tuning and the 17 representative full-data jobs remain pending; no hyperparameter selection has been frozen. Production readiness and all 423 production jobs remain unverified.

## Remaining research closure

The eight unscheduled blocks need implementations (A12, B06, B10, H08, K05, K08, M08, N06). In addition, the **122 partially covered blocks** require varying degrees of deeper implementations, exact causal controls, full reference fidelity, robust privacy/communication restrictions, task-free boundary detection, actual open-world class discovery, dynamic classifiers, higher-order causal comparisons and other planned variants.

To achieve genuine scientific closure: complete missing and partial variants; reproduce strong contemporary FCL baselines (e.g. SCAFFOLD, GLFC and appropriate peers); run the full official CIFAR experiments with multiple independent replications; verify all stage-by-stage outputs; perform negative controls and statistical comparisons; publish reproducible figures, tables, checkpoints and limitations.

For exact per-block implementation status and jobs see [`EXPERIMENT_CATALOGUE.md`](EXPERIMENT_CATALOGUE.md).
