# Diagnostic campaign release gate — 2026-11-28

## Decision

**Not released for production execution.** The validated code revision passes local and remote CI, and two Mtech CIFAR-100 GPU smoke paths plus independent artifact verification passed. The ten CF confirmation slots still lack scientific bindings; failure/recovery validation and a durable two-host coordinator remain incomplete. No diagnostic production run was admitted.

## Source and inventory

- Validated campaign code revision, pushed on `main`: `06410c7e8f6199243315c767129c6dec87f4b374` (`Fix Task-IL batch and evaluation regressions`). The clean local worktree and both clean server mirrors were verified at this revision before smoke execution.
- The canonical ledger contains 152 slots. The local and remote real-data preflight compiled 142 configured jobs and reported 10 reserved CF slots. This is configuration preflight, not GPU execution or result evidence.
- No CF config, seed, selection rule, or binding is defined for the 10 reserved slots. Keep them explicitly specification-blocked until the author supplies definitions.
- The original 53-issue audit descriptions are absent from this checkout. The register preserves 53 issue identities and summary dispositions, but explicitly does not claim independent issue-by-issue resolution.

## Validation on the current candidate worktree

- `.venv/bin/pytest -q --tb=no`: **458 passed, 335 subtests passed, 8 warnings** (238.96 s).
- `.venv/bin/pytest -q tests/test_joint_*.py --tb=no`: **70 passed, 335 subtests passed** (62.68 s).
- `.venv/bin/python scripts/validate_factorial_plans.py`: passed. It compiled 32,133 valid jobs per model, 99 method/aggregator pairs, 53,805 inputs, and 21,672 rejections for each ViT/CNN plan; current primary-probe expectations are five concurrent clients and 20 intervals.
- `.venv/bin/python scripts/audit_stream_math.py`: passed with the same 53,805 / 32,133 / 21,672 input, accepted, and rejected totals.
- The full run emitted eight warnings, including existing upstream source syntax/deprecation warnings and one test tensor-to-float warning. No test failed.
- Fresh CI for `06410c7` passed both workflows: [CIFAR-100 sequential local-task protocol](https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/46156374402) and [Joint FCL isolated validation](https://github.com/Anurag9000/Correlation-Aware-FCL/actions/runs/46156374365).

## Corrections verified in this candidate

- Non-stream arrival events now provide `class_ids` to downstream event preparation.
- Task-IL EWC/MAS importance consolidation uses active task-head output coordinates.
- Task-IL GEM/A-GEM reference gradients retain the physical forward batch while excluding padding from the loss.
- Spurious reversed-test evaluation now receives task identities, so local head columns are not treated as global class IDs.
- Synthetic Task-IL label noise now selects replacement labels from each sample’s source-task label set.
- Invalid singleton Task-IL test fixtures were rebuilt with valid multi-class schedules; historical direct Class-IL API checks are explicitly marked or isolated. Native GLFC/LGA/FEAT Task-IL pilot construction now remains blocked because those methods lack a task-head adaptation.
- Factorial snapshot fingerprints/examples and stale primary-probe expectations were refreshed without changing validated plan counts or scientific configuration values.

## Real-data smoke evidence

- Mtech `MtechDEServer1`, clean mirror at `06410c7e8f6199243315c767129c6dec87f4b374`; Tesla V100-PCIE-32GB. Initial free memory was 32,497 MiB. During the run the process used 2,712 MiB according to `nvidia-smi`; recorded peak CUDA allocated memory was 1,088,733,696 bytes.
- Runtime was Python 3.12.15, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA build 12.1. The existing Mtech `.venv` and CIFAR-100 payload were read-only inputs from the preserved older checkout; source was imported from the clean mirror.
- Smoke run `SMOKE_CIFAR100_TASKIL_GPU_20261128_MTECH`, output `/home/anurag24092/Projects/CAFCL-smoke-20261128-mtech/runs/SMOKE_CIFAR100_TASKIL_GPU_20261128_MTECH`. It used the full 45,000-sample CIFAR-100 training split, 100 labels, five clients, ten learning intervals, one epoch, 750 optimizer steps, and a final communication-only flush (11 recorded intervals total). All 50 learning uploads completed; local recorded losses were finite. Logical tensor payload totaled 2,245,954,400 bytes; this is simulator accounting, not a measured network transfer.
- The run completed with all 100 classes observed. Final smoke metrics were Task-IL test accuracy 0.6595 on 10,000 test examples and validation accuracy 0.6662 on 5,000 examples. These are smoke health metrics, not production results or a comparative finding.
- `cafcl.verify_final` independently reloaded the final checkpoint on CUDA and recomputed test, validation, and per-head Task-IL accuracy. All 104 recorded comparisons matched exactly (`verified: true`). Final model SHA-256: `9a3d477eaacbf2fdb3731b69251dfc7a9440c683c5c49172f268030a18ebb6a`; metrics SHA-256: `a9e45c4f1c5d11448833e87626c74f898d8e9543689055d97e626041ad31fe56`; verifier JSON SHA-256: `b46d0d09a1818ede7b27ac24fc76182eb765a85977e2e4a0cfa1dd2e10e7ddf3`.
- A separate stationary reference-engine smoke used clean mirror revision `3f4b8fc7c737ecfe08eae3d751359a6ae5be83eb` (the same validated code plus documentation commits): `SMOKE_FL002_20261128_MTECH`, under `/home/anurag24092/Projects/CAFCL-smoke-stationary-20261128-mtech/runs/FL/SMOKE_FL002_20261128_MTECH/attempt_0001`. It trained one epoch over all 13,500 TRAIN samples from CIFAR-100 classes 0–29, completed 211 optimizer steps, and passed `validate_artifacts`. Recorded test accuracy/loss were 0.2956666666666667 / 2.504348576863607 on 3,000 examples; validation accuracy/loss were 0.30333333333333334 / 2.489096918741862 on 1,500 examples.
- Independent stationary checkpoint inference reproduced all four accuracy/loss values exactly and confirmed the saved runtime and dataset identities. Checkpoint SHA-256: `710608984c408e59f8f9edce6962d12b0fb5865370fbdc133cba5c84f3d2aebe`; metrics SHA-256: `0c994389e994023020aa5568677fe698f61fa18c264a9ceb171634169ec5d831`; artifact receipt SHA-256: `a599ed8d7be3a440a3a2abad6c76e0cbceead747ad77004527db636db623de70`; independent verification JSON SHA-256: `0dced01d3b63938b38f7558ffa8dbf327dcf32bc18913a7b3b90eede175aeb01`.
- A post-run TRAIN-only CUDA backward check on both final checkpoints used 32 samples from each Task-IL class and two samples per each stationary class. Both losses and all gradients across 62 parameter tensors were finite. The gradient-check report SHA-256 is `ec7b8a374a0199497da95ebd9c203f74c5e7431e5f69a7295b44b3c17da07208`.
- V100-server remains un-smoked. Across five snapshots from 20:32–20:36 IST, free memory fluctuated from 9,686 to 21,178 MiB and GPU utilization from 0% to 100%; Boltz processes were active and a separate FedQCL process appeared during the window. At the 20:54 IST recheck, V100 had 25,324 MiB used, 7,176 MiB free, and 100% utilization, with three unrelated Boltz/FedQCL compute processes. Mtech at 20:54 IST had 32,497 MiB free, 0% utilization, and no compute processes. No CAFCL run was started on V100 because its GPU remained occupied and volatile. Both clean mirrors were confirmed at the current evidence-document revision.

## Remaining release requirements

1. Locate the original 53-issue audit descriptions and independently map each finding to source, tests, and ledger jobs.
2. Obtain scientific definitions for the ten CF slots or retain them as specification-blocked.
3. Complete deterministic synthetic integration across every intended production engine and supported method/aggregation pair. Existing checkpoint-write interruption, simulator resume equivalence, local exclusive-lease, hardware-gate, and incomplete-artifact rejection tests passed in the full suite; SSH interruption, broker restart, stale scheduler reconciliation, worker exit without a receipt, resource exhaustion, and separated retry-attempt behavior still lack end-to-end injection evidence.
4. Perform and independently verify a real CIFAR-100 GPU smoke on V100-server when its unrelated GPU assignments release stable capacity.
5. Implement and verify one persistent two-host coordinator with atomic claims, attempt identities, resource reservations, recovery, and artifact checksums. Both `diagnostic_campaign.run_suite` (the 152-slot ledger) and the joint broker currently schedule only local subprocesses and GPU IDs; finding `NEW-011` records this implementation gap.
