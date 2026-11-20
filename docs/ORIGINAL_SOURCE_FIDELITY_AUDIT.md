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
not the paper's `OURS` self-distillation. The active adapted CE/FedAvg runner
now executes the source `CE._train` path in no-replay mode and has a separate
Replay adapter that accumulates gradients over the full currently available
archive for each incoming batch update. It also uses the source first-k
exemplar constructor and source `FedAvg.update`. The class-index mapping,
fixed 100-row adapter, client schedule, archive sharing, and offline behavior
remain protocol adaptations; this does **not establish full trainer fidelity**.

## Explicitly permitted changes to the originals

The active paired experiment requires five persistent clients; one adjacent
two-class task/client at each of 20 learning syncs beginning at sync 1; random
selection of one client per sync whose model upload is withheld while all five
train; a communication-only final flush; full archive delivery to every
client; and class-IL inference. The two arms run as separate concurrent
processes, each with five client workers and a server model. The requested
batch, LR, epoch and memory settings are also user-specified variables.

Older v6–v22 stream descriptions and factorial manifests retain their original
settings as historical or separate research designs; they do not define the
current primary paired runner.

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
\n## Byte-for-byte vendor provenance\n\nThe pinned `docs/ORIGINAL_UPSTREAM_BLOB_MANIFEST.json` records the Git blob SHA-1, size, executable mode and source commit for **all 76 original files** across the four repositories: GLFC 17, LGA 16, FEAT 18, FedCLASS 25. `tests/test_original_upstream_blobs.py` verifies every file against its Git blob SHA-1 during CI. This proves source-copy integrity only, **not original algorithm execution**.\n
## Actual original aggregation now executed

The four paired pilot execution paths now dispatch aggregation directly through
`src/cafcl/original_aggregation.py` using author-source code:

- GLFC: `GLFC/src/Fed_utils.py:FedAvg`, uniform original averaging.
- LGA: `LGA/Fed_utils.py:FedAvg`, uniform original averaging.
- FEAT: `FEAT/utils/dmc.py:average_weights`, uniform original averaging.
- FedCLASS CE: `FedCLASS/utils/fed_algorithm.py:FedAvg.update`, original sample-size weighting.

The published original function bodies are compiled directly from their
original AST in the first three cases; FedCLASS's original class is imported.
Original source is unmodified. The upstream aggregation runs on the **accepted**
model candidates. For delayed offline updates, candidate construction remains
the explicitly noncanonical version-lag-damped and BN-safe policy; it is **not**
original-paper stale-update behavior. This is a real core-mechanism fix but it
DOES NOT yet restore missing local/proxy/geometry/memory machinery or certify
any pilot as fully faithful.

## Original weighted BCE source now invoked (GLFC and LGA)

The production local-loss path in `src/cafcl/native_methods.py` now calls
`src/cafcl/upstream_losses.py`, which compiles and executes the exact
`_compute_loss`, `efficient_old_class_weight` and `get_one_hot` function
bodies from the author's immutable source files, with the sparse class IDs
converted into contiguous coordinates. This replaces the earlier hand-written
formula approximation, including LGA's historical task-dependent weight
modulation and clamping. The old-teacher output is selected in old-class-first
order as assumed by the original sources. The local training remains CUDA-only
for these methods, as in the original code. This **does not** restore or certify
original `beforeTrain`, entropy detection, exemplar memory, prototype gradients,
proxy reconstruction or LGA geometry. The original modules are not modified.

On CPU synthetic tests only, the source-AST loss bridge changes the
original device transfer syntax `.cuda()` to `.to('cpu')`. All mathematical
source nodes and original GPU training code remain unchanged. The normal
paper-fidelity production pilot still requires CUDA.

## Original FEAT relation loss now executed

The `native_feat` local objective now calls the exact original
`FEAT/methods/FEAT.py:gsa_distill_loss` body via `original_feat_gsa_source`.
The integration maps sparse class IDs to contiguous labels and transposes the
standard Linear classifier's `[C,d]` weights into the expected `[d,C]` form.
This is a **protocol/model-shape adaptation**, not proof of original FEAT
network parity. The official `IncrementalNet_CausalETF` remains absent, as do
the original retention selection and inference correction in the active runner.

## FedCLASS CE source training and exemplar selection execution

`src/cafcl/upstream_fedclass_ce.py` now executes the author's original
`FedCLASS/models/ce.py:CE._train` via original source AST, including SGD
recreation and AMP scaler behavior. For user-specified 50/class memory, it also
executes `BaseLearner._construct_exemplar_unified`, selecting the first
eligible examples per local class as the published CE source actually does.
This **replaces** the previous Algorithm R reservoir for the FedCLASS CE pilot.
The original `FedAvg.update` is now used for aggregation. The old CLI string
`--memory-mode reservoir` remains for backward compatibility, but the manifest
now states the actual original selection policy. Source `FlTrainer.fl_train`
and dynamic classifier growth remain protocol adaptations, not original
function calls. The method is still **NOT certified fully paper-faithful**.

**FEAT source-gradient parity:** `gsa_distill_loss` explicitly detaches the
prototype-relation target `logPP`. Thus GSA gradients flow into student
features but NOT into the target prototype/classifier weights. The regression
tests assert this original behavior (rather than incorrectly demanding a
prototype gradient).

## 2026-11-20 source-method experiment identities

GLFC and LGA use `PILOT_SOURCE_CORE_V4` job IDs and `seedN/source_v4`
outputs. FEAT uses `PILOT_SOURCE_CORE_V5` and `seedN/source_v5`. FedCLASS
uses its separate `SOURCE_V3` identity. It now has explicit
`authors_exemplars` and `none` modes, uses `SOURCE_V3` job IDs, and has
`*_source_v3` output folders. The old FedCLASS CLI flag `reservoir` is
rejected so its source first-k exemplars cannot be mislabeled as Algorithm R.
Its original CE optimizer resets; there is no source 50/50 replay ratio.
These identities prevent silent reuse of earlier completed partial ports.

**No change to the fidelity gate:** all four are still incomplete until
all original mechanisms and per-stage source calls are implemented.

## FEAT ETF geometry reconstruction (NOT author-source parity)

The source-backed FEAT pilot now uses the original vendored FEAT ResNet18
feature extractor paired with a deterministic unit-norm simplex ETF classifier
rather than a trainable linear classifier. This head is reconstructed from the
CVPR 2026 paper's fixed-ETF geometry because `utils/inc_net.py` is absent from
the authors' official repository and its git history. An orthonormal embedding
gives all off-diagonal prototype inner products exactly `-1/(C-1)`; the
prototypes are frozen parameters, shared by model state. The source GSA function
is still executed.

The output is marked `paper_reconstructed_fixed_ETF` and **not** the
unpublished authors' `IncrementalNet_CausalETF`. FEAT V5 job IDs distinguish
the fixed-ETF model from the earlier trainable-head native-loss pilot.
The inference energy correction and exemplar scheduling are still missing.

## FEAT ECC original source execution and synchronization

`src/cafcl/upstream_feat_geometry.py` now compiles and executes the **original
source** projector, energy estimator and cross-client ECC-prior aggregation.
The native FEAT local-training path records each client's actual old/new
feature-space energy priors at the end of an experience; those priors persist
through disconnected intervals and reach the server **only** when the client
uploads. The server then uses the original `aggregate_ecc_priors_from_clients`
formula and publishes the accepted result to connected clients. Projection
matrices, scalar energies and source receipts are checkpointed as FEAT-specific
state; hidden client information is not advanced into the server prematurely.

**Limitations:** The authors' FEAT ETF inference implementation
(`utils/inc_net.py`) is absent upstream. The projection and scalar priors are
now actually computed and synchronized but cannot be called a faithful final
inference correction until that missing original model path is recovered. The
required fixed physical batch 64 is implemented using the shared padded tail
sampler and may repeat tail examples in ECC EMA statistics, a user-protocol
adaptation to flag in reported results. Full-method certification remains
**FALSE** and the launch gate remains closed.

## FEAT geometry applied at inference (third-party reconstruction)

`src/cafcl/feat_inference_correction.py` now applies the FEAT projector
energy-gating correction to actual class-incremental server validation and
test predictions, using ONLY accepted-and-synchronized ECC statistics.
The formula is independently reconstructed from
https://github.com/kiennkt05/FEAT/blob/main/utils/inc_net.py because
the official author repository does not include `utils/inc_net.py`.
This is **not original-author source** and is NOT certified as equivalent
to the missing code. The source-fidelity gate remains closed.

## FEAT: author's FLDataSelector now invoked for exemplar retention

`src/cafcl/upstream_feat_replay.py` executes the unmodified author
`FLDataSelector` AST, including random QR masking, singular-value leverage
scores, aggregation of probabilities and multinomial **with replacement**.
The pilot now uses that author-selected memory rather than Algorithm R for
FEAT and retains duplicate selection tickets for local replay. The common
server archive deduplicates image payloads in transit.

**Protocol deviation:** The original FEAT selects jointly across all client
feature sets at a global task boundary. This five-client independent task
stream needs private selection even when clients are offline, so the current
bridge performs class-local/source-functional selection. Its numerical
selection distribution differs from the original joint cross-client query
allocation. This is NOT yet full FEAT fidelity. Remaining FEAT gaps:
`FEAT._select_data_for_retention` joint call, complete original ETF module
unavailable, EGC correction at inference, local original
`_local_finetune` class-aware training, and original replay composition.

## FEAT training-time original transform source execution

The FEAT pilot now invokes `FEAT._get_norm_and_transform('cifar100')`
compiled directly from unchanged upstream `FEAT/methods/FEAT.py`. The
resulting original RandomCrop(32,pad=4), RandomHorizontalFlip,
ColorJitter(brightness=63/255), ToTensor and Normalize apply to every
FEAT local TRAIN/replay batch. This restores source-specific training
augmentation missing from the generic pilot, while TEST and validation
remain normalized only. Input bytes are recovered from affine-normalized
CIFAR100 examples before applying the actual author PIL transforms.

The experimental worker uses a per-client deterministic CPU RNG scope to
adapt authors' sequential random augmentation to concurrently scheduled
clients. This changes RNG consumption order (not transform definitions)
and must be documented. FEAT receives a fresh `SOURCE_CORE_V5` job
identity and output directory; no older experiment artifact is overwritten.
Other original FEAT mechanisms still remain incomplete and fidelity is
still NOT certified.

## GLFC and LGA source entropy/herding in the pretraining path

`src/cafcl/upstream_glfc_lga_memory.py` now calls original-source
`update_new_set`, `entropy_signal`, `_reduce_exemplar_sets` and
`_construct_exemplar_set` on every local task BEFORE training, including
disconnected clients. It persists original per-client entropy baseline,
last-class pair, learned-class order, and selected exemplar indices.
Original CIFAR RGB arrays are mapped back to immutable per-client TRAIN
index records. It no longer inserts generic Algorithm R memory for these
two methods. Server uploads include only **actual source-selected exemplars**
from accepted clients.

The source authors choose 6 classes or `args.iid * task_size` in
`beforeTrain`, which cannot execute unchanged when the user requires
exactly 2 new classes per client. An explicit current/last-two-class
adapter is used instead. The author's `memory_size / learned_numclass`
calculation is preserved but the total source budget is set to
user-specified 50 per learned class. The original `compute_class_mean`
preprocessing and feature extraction are adapted into fixed physical
batch-64 inference (rather than GLFC's whole-class forward and LGA's
50-images batches), keeping the original feature-normalized mean and
unmodified original herding argmin. These are recorded as
**protocol-required differences**, not hidden as bitwise reproduction.
Entropy triggers may legitimately be false, in which case no new class
exemplars are stored, exactly as source `update_new_set` prescribes.

**Still missing:** original source proxy gradient generation,
gradient inversion/reconstruction, original model-back teacher decisions,
and LGA's radius/geometry monitor. Full-method fidelity gate stays
**CLOSED**.

## Author-specific continual state merged and downloaded

Server synchronization now archives origin-tagged method-specific artifacts
from **accepted client uploads only**, including GLFC/LGA entropy history,
learned-class/exemplar-selection ledger, and FEAT's unmodified
leverage-selection tickets/receipts. Other clients receive exact copies of
the acknowledged ledger after synchronization; offline client-private
states are NOT revealed until reconnection. Download verification checks
this receipt too. It deliberately does **not overwrite another client's**
local task-trigger state with a peer's entropy baseline: the original
per-client method remains local and the server retains the union as
origin-provenanced history, consistent with each source's semantics.
This doesn't replace the original proxy reconstruction mechanisms.

**Source-trigger regression:** CPU synthetic tests now force a threshold
crossing and verify that exactly the original source herding chooses images
from the PREVIOUS pair only, without reading future or other-client examples.
The ordinary two-pair local-training test checks that a non-triggering entropy
signal does **not** produce fake Algorithm-R exemplars.

## GLFC/LGA original source train-time augmentation

`src/cafcl/upstream_native_augmentation.py` now compiles the original
`train_transform` assignments from official GLFC `src/fl_main.py` and
LGA `fl_main.py`, including exact RandomCrop, RandomHorizontalFlip,
ColorJitter, ToTensor and CIFAR100 Normalize. The authors' PIL transforms
run on actual available TRAIN/replay examples in the local BCE objective
and on the current examples used by original `entropy_signal`.
This is the original transform code, with a deterministic per-client RNG
scope and fixed 64-size physical model forwards. Validation and TEST
remain unaugmented. GLFC/LGA pilot identities advance to source V4;
older V2/V3 results are not overwritten. Source training optimizer and
proxy-server reconstruction remain incomplete, fidelity gate CLOSED.

## Original LGA radius calculation

`src/cafcl/upstream_lga_radius.py` compiles and executes the **exact
original** `LGA/myNetwork.py:network.compute_radius` and
`network.Image_transform` bodies at the second class-stage boundary.
The source covariance diagonal/trace and root-mean-radius arithmetic
is unmodified. The DATA are only server-acknowledged TRAIN examples
from original classes 0 and 1. Their number is 450/class instead of
the upstream original 500, so the source is given 9 instead of 10
fifty-image slices; each underlying model feature forward is padded to
the user's physical batch64. The scalar radius is part of committed
federated CL state and is sent only to synchronized clients.

This does NOT yet invoke original `LGA/ProxyServer.py:f_monitor`,
so the estimate is not sufficient for full algorithmic fidelity.

## Source-executed original GLFC/LGA local training

The two GLFC and LGA experiment runners now dispatch to
`src/cafcl/upstream_glfc_lga_train.py`. It invokes **the exact original
source** `GLFC_model.train` / `LGA.train` method bodies compiled from the
immutable author files, with their own fresh SGD initialization, no
momentum, original weight decay `1e-5`, and source local-epoch
learning-rate schedule. These replace the generic persistent-SGD loop.
Original `_compute_loss` and `efficient_old_class_weight` also execute
inside source training. The authors' original `update_new_set`
builds the actual current+old-exemplar local loader; accepted historical
server replay is added solely as demanded by the user's synced-class-IL
protocol. Each model still trains on physical batch64. The original
gradient-generating proxy learner, original server `model_back` selection
and LGA proxy `f_monitor` remain missing, so the interim historical
teacher is explicitly labelled noncanonical. **Not fully faithful.**

## FEAT published source local SGD and GSA orientation defect

`src/cafcl/upstream_feat_train.py` invokes the actual unchanged source
`FEAT._local_update` for the first pair and `FEAT._local_finetune`
for subsequent pairs. It uses the original SGD recreation, momentum=0.9,
source weight decay, CE+0.1·GSA objective and source ECC estimation
with published `ema_decay=0.8`. The shared generic FEAT optimizer path
is bypassed. The source's own teacher is `eval()` only and unused in its
loss (this is confirmed by the original source).

The **published FEAT source itself has a shape error**:
`FEAT._local_update/_local_finetune` passes `model.fc.weight` in
`[C,D]` layout, yet `gsa_distill_loss` expects prototypes `[D,C]`.
The source would fail when executing the original ResNet+classifier
without adaptation. We do **not** rewrite the original function;
the compatibility namespace transposes the matrix at this one boundary
while executing the unchanged original GSA math. This is NOT bitwise
author-source behavior and is explicitly recorded in the per-task output.
The FEAT ETF head and geometric correction remain unverified
paper/third-party reconstructions. The original `_fl_train` aggregation
loop still needs protocol integration; the fidelity gate stays CLOSED.

## Original GLFC/LGA auxiliary-encoder prototype gradients

`src/cafcl/upstream_native_prototypes.py` now executes the official
`proto_grad_sharing`/`prototype_mask` source bodies **when**
the unchanged author entropy condition is true. It constructs the exact
original auxiliary LeNet architecture and `weights_init` from the
immutable author files, then applies the original 50-step prototype
input optimization and extracts the source CE encoder gradients.

Sparse client class coordinates are mapped by permuting the ORIGINAL
100-way encoder output logits (not changing the CE denominator),
so `ProxyServer.gradient2label` can infer the true global class
from the original last encoder weight gradients. Source model forwards
are padded to physical batch64 then sliced back to their logical
single-prototype input, preserving the original single-example loss
gradient. This is a disclosed user-protocol input-shape adapter.
Only client-local already arrived TRAIN images are used.

Generated gradient receipts are stored during hidden local execution
and enter the server acknowledged gradient archive **only** on actual
upload. Synchronized clients download their union and verify source
gradient tensor identity. An entropy trigger that never fires generates
NO prototypes, exactly as the source prescribes. Note that threshold
`1.2` was calibrated to different class increments; it may remain
dormant under 2-class local additions. The original server
reconstruction/monitor/model_back is still outstanding. The complete
pilot is NOT yet fully paper faithful, gate CLOSED.

## GLFC/LGA original proxy component isolated — not yet server-connected

`src/cafcl/upstream_native_proxy.py` extracts the published source
`proxyServer.dataloader`, `gradient2label`, `reconstruction`,
`monitor`, LGA `f_monitor`, `model_back`, plus the original
`Proxy_Data` class. These are executed from the authors' unchanged
source AST; a source-compatible original LeNet encoder is seeded
identically to the client prototype-sharing encoder. The original
250-step LBFGS inversion and twenty reconstructions per label are
retained in source `reconstruction`; no substitute algorithm is
claimed. Tests verify original entrypoint provenance, gradient-label
identification and 64-sample monitoring.

**Integration status:** the actual simulator's GLFC/LGA server still
does **not** invoke `proxyServer.dataloader` or apply `model_back` to
subsequent local teachers. This component by itself does not close the
source-to-runtime gap. Source `dataloader` assumes sufficient gradient
prototypes to make at least one 64-sample monitoring batch even with
`drop_last=True`; two-class stages may not. This incompatibility must
be handled explicitly, without claiming original training semantics.
Fidelity gate remains CLOSED.

## Source proxy attached to accepted-only global server path

`src/cafcl/upstream_native_proxy_server.py` now contains a production
step that runs original-source `proxyServer.dataloader`,
`reconstruction`, `monitor`/LGA `f_monitor`, and `model_back`
after *accepted* model/gradient uploads, plus an explicit
`original_source_teacher_slots` builder for subsequent original local
`GLFC_model.train`/`LGA.train` calls.

The two-class protocol can produce only 40 reconstructed images from two
prototype receipts. The original proxy's `drop_last=True` loader of
physical batch64 cannot monitor these. Rather than perform invalid
zero-sample monitoring, the adapter accumulates accepted gradient receipts
until >=4 are available. This is an **explicitly non-original** protocol
compatibility rule; exact per-sync source behavior cannot be claimed. The
original source 250 LBFGS inversion iterations and 20 images/prototype
remain unchanged whenever enough receipts exist. No hidden client's
receipt can enter the backlog before that client reconnects.

## GLFC/LGA original proxy server now in the *actual* training run

The `simulator.run_experiment` accepted-upload branch now invokes
`source_proxy_server_step` after accepted client model/prototype
merging and before any client downloads. It invokes actual original
`ProxyServer.dataloader`, LBFGS `reconstruction`, monitor/LGA
`f_monitor`, and `model_back`, when the source has sufficient
accepted reconstructed gradients. New and buffered accepted source
prototype receipts are the **only** inputs; unsent offline-client
gradients cannot leak to the server.

The original `GLFC_model.train`/`LGA.train` call now receives
**two separate, source-selected historical teacher snapshots** from
the last acknowledged server broadcast, not the former fabricated
`[local_teacher,local_teacher]` pair. The original entropy signal
continues to choose which slot to use.

The source monitor's `drop_last=True` batch64 assumption is incompatible
with a single 2-class task that reconstructs only 40 proxy images.
The protocol explicitly buffers accepted gradient receipts until
at least 4 prototypes produce at least 80 reconstructions. This
delayed proxy-monitor trigger is **not from the original papers**.
It must be treated as an experimental setting adaptation, never
as proof of exact original synchronized per-round behavior.

No GPU/full CIFAR-100 source proxy inversion has been run by CI;
CI exercises the original source entrypoints and mocked orchestration
but cannot certify 250-step, multi-prototype CUDA runs or original-paper
accuracy. The independent scientific-fidelity gate remains CLOSED.

## Original Proxy_Data source NumPy-array compatibility

The original `Proxy_Data.getTestData` concatenates samples into an ndarray
but its original `__len__` and `__getitem__` test
`self.TestData != []`. Modern NumPy cannot broadcast an array of image
tensors against an empty Python list and raises `ValueError`. The
separate `proxy_dataset_adapter_class` executes original `getTestData`
unchanged, then converts the source ndarray to a list with a
`.shape[0]` property. Thus both original downstream accessors also
execute unchanged. This is explicitly recorded as an upstream
compatibility repair, not a claim that the original repository runs
unmodified. Original source blobs are still immutable.

## LGA original initial proxy radius

The LGA source network initializes `radius=0.15`. The proxy's compatible model now uses that original default before the source covariance-radius update, rather than incorrectly forcing 0.0 (which would select the plain rather than the geometric `f_monitor` branch). The new conditional state remains an experimental compatibility layer, not a change to any original vendored file.
