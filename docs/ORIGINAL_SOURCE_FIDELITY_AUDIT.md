# Mandatory source-to-runtime fidelity repair — GLFC, LGA, FEAT, FedCLASS

**As of 2026-11-20: No existing pilot is a complete original-method reproduction.**
Original source trees remain untouched. Prior experiments are preserved as
historical **adapted-loss ablations**, not renamed as faithful paper benchmarks.
This document deliberately distinguishes *source file present*, *native class
invoked*, *numerical parity*, *full algorithm coverage*, and *reproduced results*.

The executable machine-readable capability inventory is
`src/cafcl/upstream_fidelity.py`. Every mechanism listed there is currently
marked **adapted, imported only, omitted, or absent upstream**. No algorithm
can be certified by a unit test that only checks 100-class logits or a
ResNet import.

## Why the previous pilots were wrong

The previous main-path `src/cafcl/native_primary.py` and
`src/cafcl/fedclass_primary.py` both dispatch through
`src/cafcl/simulator.py` and `src/cafcl/continual.py:local_train`.
The former three ultimately call `src/cafcl/native_methods.py`, a small set
of replacement losses, **not** the original GLFC, LGA, or FEAT learners.
FedCLASS uses generic CE/ER and never executes `FedCLASS/models/ce.py:CE._train`.
The source files are copied but method-specific code is dormant.

The GLFC/LGA original frameworks use entropy-triggered memory,
feature-space herding, prototype-gradient generation, gradient-reconstruction
proxy training, proxy-guided teacher selection and source-specific aggregation.
The current pilots omit those routines, use a shared global raw-image reservoir,
and even replace the original uniform FedAvg with sample-weighted FedAvg.

The original LGA additionally uses task-aware example weighting, radius
estimation, geometric augmentation and `ProxyServer.f_monitor`, none of which
execute in the current pilot.

Original FEAT implements FLDataSelector's feature masking, QR/SVD leverage
scores and data selection, GSA, the geometry-aware `IncrementalNet_CausalETF`
head, projection matrices, energy priors and the correction mechanism. The
existing pilot uses only an adapted GSA objective and a conventional head.
Moreover the authors' [qizhuang-qz/FEAT](https://github.com/qizhuang-qz/FEAT)
main tree (SHA `459bab31e19a4215bd1cadde22296507d9f3fdb6`)
is itself missing `utils/inc_net.py`, `methods/GSA.py`, and
`methods/EGC.py` while importing them. One distinct
[third-party reconstruction](https://github.com/kiennkt05/FEAT)
contains a compatible ETF network, but that is **not automatically the
authors' missing source**. Its provenance and equivalence must be established
before use; its files must not be quietly represented as original.

For FedCLASS, the user originally selected the **CE + FedAvg baseline**,
not the paper's `OURS` self-distillation. Nevertheless the source CE learner
has optimizer resets, AMP, an incremental `CENet`, client-local class selection
and its own memory construction. Our current adapted CE/FedAvg runner instead
uses persistent SGD, fixed 100-row logits and a generic replay buffer. Fresh
sample-weighted FedAvg arithmetic can agree numerically, but this **does not
establish trainer fidelity**.

## Explicitly permitted changes to the originals

The experimental intervention requires five persistent clients; the
initial two sequential 2-class tasks per client followed by one 2-class
task per client per U2–U9; 9 training synchronizations plus a
communication-only U10; FULL 5/5 uploads/downloads; OFFLINE 4/5 communication
while 5/5 train; terminal delivery of accumulated hidden learning; class-IL
inference. The requested batch, LR, epoch and memory settings are also
user-specified experimental variables.

These are **protocol adaptations** and must be isolated in adapters and
recorded. Original local losses, optimization semantics not explicitly
overridden, memory-selection algorithms, network/classifier behavior,
teacher/monitor/proxy and geometric components, server computations,
and their associated numerical semantics must otherwise remain intact.

"FedAvg" also has different exact semantics across sources: GLFC/LGA/FEAT
average source-model weights uniformly; the FedCLASS CE baseline averages
by client training-example counts. A faithful port must not force the same
implementation simply to simplify benchmarking. Where an offline client
requires a genuinely new stale-update rule, isolate and label it as a
user-protocol extension, never claim it comes from the paper.

## Exact completion criteria for each suite

1. Implement a new source-backed runner that **calls** the original method
   functions. Keep the originals byte-for-byte immutable. Have tests spy on
   actual execution of original entry points, not only matching mathematics.
2. Feed the prescribed sparse client class streams through a clearly scoped
   dataset/class-index compatibility adapter, including per-client training
   and historical state. No test split or future-class leakage.
3. GLFC: execute original entropy detection, weighted BCE teacher
   distillation, exemplar herding, `proto_grad_sharing`, proxy
   gradient-label recovery and reconstruction, monitoring and teacher return.
4. LGA: execute all corresponding original methods **plus** radius,
   geometric feature monitoring and task-dependent sample reweighting.
5. FEAT: recover/reconstruct the missing source modules with strong
   provenance; execute original GSA, leverage-score masking/selection,
   ETF classifier, geometric projectors, energy-prior aggregation and
   inference correction.
6. FedCLASS CE baseline: execute original `CE._train`, original
   `CENet`/classifier growth and original `FedAvg.update` with protocol
   adaptation, including its actual optimizer/memory behavior. Do not swap
   to `OURS` unless separately requested.
7. Preserve all original algorithmic state during offline windows.
   Reconcile model dimensions under independent client class arrivals;
   merge stale contributions and mechanism-specific metadata without
   inventing unsupported vanilla FedAvg equivalence.
8. Add source-code provenance, deterministic stream equivalence checks,
   **behavioral/numerical parity tests** for original small fixtures,
   proxy/reconstruction tests, memory-selection tests, geometric tests,
   reconnection/terminal U10 tests, full resumption tests and a CUDA smoke.
9. Run the two-arm tests and final CIFAR-100 GPU suites; publish honest
   class-IL accuracy, per-class metrics and precise deviations. Do not
   present a partial protocol port as reproducing the papers.

**Fail-closed release policy:** default production pilot runners now refuse
to execute as though they were original-paper faithful. They still permit
`--plan-only`, and the explicitly named `--permit-adapted-legacy` escape
hatch can execute an **adapted-loss ablation** for historical comparisons,
not the original paper. The 8-arm campaign supervisor also enforces this
before allocating GPU memory. Previously launched external runs and archived
experimental evidence are not modified.

**Current status:** The full source-faithful rebuild specified above is NOT
complete. The gate and audit are a safety correction, not an assertion that
the missing original mechanisms have now been implemented.
