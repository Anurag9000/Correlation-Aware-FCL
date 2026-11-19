# v14 — Correcting measured CIFAR-100 classifier collapse without changing FCL stream

## Motivation and observed evidence (historical v10, not v14 accuracy)

The finished single-GPU v10 dual-ResNet-18 job stored in
\`results/primary_cifar100_single_gpu_parallel_v10_cnn_pct11p1111_bs64_seed0/\`
reported 4.62% full-participation versus 5.64% rotating-offline final 100-class accuracy. Its final confusion matrices indicate 78/100 and 73/100 labels were **never** predicted respectively. After U1, local task losses frequently reached 4–5. The pilot's evaluations were terminal-only, so these values could not distinguish acquisition failure from post-FedAvg degradation.

Those are historical observations, NOT evidence that the new v14 code has improved accuracy. GPU training has not been run for v14.

## Verified mathematical flaw in the previous local Replay objective

Under strict two-class episodes, the old v13 local loss was
\`CE((z_a,z_b),y)\` for *new* data, but \`CE((z_0,...,z_99),y)\`
for replay data, while every inactive classifier row was frozen.

The new-stream pairwise softmax is invariant to a common shift in the
two current logits. The 100-way historical replay CE is NOT invariant to
that shift; when its correct label is old, its gradients on both
current-class biases are positive, systematically lowering both biases.
This can make a pairwise classifier work while rendering the new labels
incapable of winning the global 100-class argmax.

### v14 fix — class-incremental seen-class competition

All registered methods keep the EXACT SAME incoming two-class task events,
9 training intervals + one terminal flush, online/offline rules,
class-ID mapping, one shared 100-class linear layer, and class-IL evaluation.

For methods other than ER-ACE, each new incoming example uses
\`cross_entropy(logits[all_classes_currently_known_to_client], target)\`.
Those classes are precisely the union of (a) classes received in prior
accepted global CL snapshots, (b) locally observed classes, and (c) the
new two-class task. Classes not yet observed by that client CANNOT compete.
All **known classifier rows** are trainable. Only truly UNSEEN future
classifier rows are gradient-masked, restored after the optimizer step,
and removed from momentum/weight decay changes.

For Replay, historical-memory CE uses the exact same class universe.
Historical exemplars are taken from the already-synchronized global
50-example-per-class archive plus local private reservoir.
This is critical: updating old output rows can now improve alignment
with the continuously updated shared backbone instead of preserving
obsolescent old weights forever.

For GEM and A-GEM, frozen future rows are masked from current/reference
gradients and projected updates; known historical rows are allowed
to change, with the algorithm's gradient constraints still active.
EWC, SI, MAS keep their genuine quadratic importance penalties while
allowing regularized past parameters to move, rather than secretly
enforcing an extra rigid frozen-head barrier. LwF retains old-logit
distillation. DER/DER++ retain pre-step stored logits and old-history
penalties; DER++ replay CE is now computed on known labels.
**ER-ACE is intentionally asymmetric:** its *current* stream CE remains
restricted to the new class pair, as per the core ER-ACE mechanism;
its historical supervised memory term uses all seen classes.
Thus its retained distinctive mechanism must not be conflated with
symmetric Replay.

This is a **user-setting-compatible CL objective adaptation**. Original
papers used different output architectures and tasks; an assertion of
bitwise equivalence to their entire experimental pipelines is not made.
No relaxation of the class stream, no task-ID oracle at inference,
and no future-label supervision is introduced.

## Momentum SGD and sufficient optimizer steps

* Default optimizer for ALL active local learning methods: SGD with momentum
  \`0.9\` and weight decay \`0.0005\`. This is NOT Adam and not FedAdam.
  The server algorithm remains the selected FedAvg/FedAvgM/FedAdam or
  custom method, unchanged. User-requested experiment variants can still
  explicitly change allowed optimizer hyperparameters as a documented ablation.
* The main two-process pilot defaults to effective batch **64** rather
  than the prior 1024. One two-class CIFAR-100 experience contains about
  900 distinct TRAIN images. At batch 1024, 10 epochs provided only
  roughly TEN local optimizer updates; batch 64 yields ~150 updates.
* The primary pilot's learning rate defaults to \`0.03\` (previously 0.003),
  with 10 local epochs; microbatch defaults to 8 (previously 4).
  Gradient accumulation preserves the effective logical batch. The
  smaller physical ResNet BatchNorm batch remains a limitation:
  BatchNorm is computed on the microbatch, **not** on a virtual
  batch of 64. Global BN buffer drift is now reported separately.
  Adjusting microbatch to 16 if GPU VRAM permits is a reasonable
  hypothesis, not a proven repair.
* Default private reservoir for the **dual replay pilot** is
  \`1/9\` of the 450 eligible training examples/class, exactly
  \`ceil(450/9)=50\` unique images/class. No validation/test examples enter
  the reservoir; the received server archive is an unbounded historical
  union, **not** a 50-example cap on all received classes.
* The primary and full factorial **default** effective batch is 64.
  The prior 1024 batch can still be requested explicitly for studies.

## Backbone labels and GPU feasibility

* Active \`vit\` alias now resolves to **standard ViT-Base/16**
  (\`vit_base_patch16_32\`): 12 Transformer blocks, 768 hidden channels,
  12 attention heads and patch size 16. Images remain **native 32×32**,
  with four 16×16 patches plus CLS, and **random initialization**.
  This is NOT a pretrained ImageNet model, nor a DeiT/ViT-Tiny model.
* \`cnn\` remains the CIFAR-stem **ResNet-18**, 4 [2,2,2,2]
  residual stages, no original ImageNet stem maxpool, randomly initialized.
  The user message ended after "replace ResNet-18 with", without
  identifying the requested replacement. It is not safe to invent a
  different CNN architecture.
* The fixed concurrency model remains two independent run processes
  sharing GPU 0, five independent client model instances/process,
  and local task barriers. Ten ViT-Base models across two processes are
  MUCH larger than ten ViT-Tiny models. A single 12/16 GB GPU might
  OOM; microbatch 8 does not guarantee feasibility. **VRAM success is
  unverified** and no silent sequential fallback is introduced.

## Per-server-sync tables for EVERY run and suite

Immediately after **every** committed communication interval, INCLUDING
the terminal communication-only flush, the simulator evaluates the
committed server model over ALL classes that have arrived anywhere
so far. The masked class-IL argmax competes among those arrived classes;
classes never introduced are not candidate labels.

The per-sync reporter writes under
\`results/<suite>/runs/<job>/per_sync_accuracy/\`:

* \`sync_001.png\`, \`sync_001.md\`, \`sync_001.csv\`, \`sync_001.json\`
  for stage U1, then 002, ... up to **sync_010** for the normal
  five-client 100-class pilot (U10 is the closing upload/download flush).
* Each report lists every client, that client's first-introduced classes
  in each earlier update, and the **current** global model's accuracy on
  those classes. The row/cell scores are NOT historic local snapshots
  and NOT training accuracy. Re-evaluating earlier classes at U2–U10
  makes forgetting visible rather than incorrectly claiming 0 forgetting.
* For rotating-offline participation, classes still privately held by
  a disconnected client are marked **PENDING upload**. Both the original
  "all-arrived" protocol score and the separate **accepted-only server
  score** are recorded; these cannot be conflated.
* JSON provides the full per-class table and predicted-class histogram,
  never-predicted labels, signed true-vs-maximum-competitor logit margins,
  global classifier-row bias values and old/new bias gap. Round metrics
  additionally record BN running-buffer client/server discrepancy and
  classifier bias drift. These distinguish representation/BN, classifier
  calibration and forgetting mechanisms.
* Held-out TEST evaluations are **audit-only**. Their scores, per-class
  results, logits, confusion matrices and bias statistics are forbidden
  from feeding training, replay selection, client selection,
  hyperparameter control or architecture adaptation. Validation-only
  scores are separately available for any permitted selector.
  Repeatedly inspecting TEST results manually for tuning is still
  a statistical multiple-testing risk; a clean final external test
  is desirable for publication.
* The official terminal accuracy continues to use the full 100-class
  test metric. Early reports are masked to *arrived* classes, so an
  early higher score must not be compared numerically with a final
  100-way score as if the tasks were equally difficult.

## Federation and scientific integrity

Canonical FedAvg continues full-model **number-of-new-training-examples
weighted averaging** over all floating-point model parameters (backbone,
classifier weights, and bias). The older v10 class-row overwrite is
NOT reinstated. FedAvgM and FedAdam continue their independently
tested server updates. The server still acknowledges and redistributes
all applicable cumulative CL artifacts; offline clients remain out
of synchronization until reconnecting.

Only the local learning objective/head-trainability, optimizer
configuration, active ViT architecture, diagnostic reporting and
checkpoint identity were changed. No server-side backpropagation on
validation/test data is introduced. Server BN buffer averaging remains
as implemented under the selected server method, and its potential
mismatch is now instrumented, NOT claimed resolved.

The v14 protocol version and output roots are different from v10–v13.
No old accuracy reports may be attributed to v14.

## Verification scope and unresolved empirical checks

Source-level and regression tests establish:
1. Old-class logits compete with the incoming class on new images.
2. Gradients and optimizer updates may modify already-known rows.
3. Truly future rows remain exactly frozen, including SGD momentum.
4. All client/server methods use the shared task schedule and receive
   cumulative history according to participation constraints.
5. Every sync writes a static per-class table and classification-collapse
   diagnostics, with terminal-only official score retained.
6. Canonical server aggregation does not receive method-specific row fusion.

**Not yet established:** CUDA feasibility of 2 × 5 concurrent ViT-Base
instances, numerical convergence under the revised hyperparameters,
final accuracy improvements, or comparisons over multiple seeds.
CPU CI passing cannot establish those. New results require fresh v14
experiments.
