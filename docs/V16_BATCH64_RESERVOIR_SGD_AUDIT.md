# v16: Historical physical batch 64 and persistent SGD audit

> **Historical primary settings:** The current primary Replay/FedAvg recipe is [PRIMARY_PROBE.md](PRIMARY_PROBE.md). This v16 audit is retained as historical methodology and does not define a current launch.

## Explicit requirements retained

- CIFAR-100 input is native 32×32. Production models are randomly initialized standard ViT-Base/16 and CIFAR-stem ResNet-18; model definitions, fixed pilot class-ID ordering and optimizer coefficients remain unchanged.
- The paired probe is **exactly two simultaneous processes forced onto GPU 0**. Each run has five independently trained clients and a separate server model. One arm participates fully; the other keeps one rotating offline client for **all nine training/synchronization intervals**, then all clients communicate at the tenth, terminal communication-only flush.
- The first local interval learns two successive two-label tasks per client (four classes); later intervals learn a single two-label task per client. Full CL-state transfer happens only at real server synchronization.
- All active CIFAR-10/CIFAR-100 production training uses **one real batch of 64 images per optimizer step**. There is no smaller model-training forward plus accumulated gradient. The old split-batch configuration field and launcher option are removed, not merely set equal to 64.
- All local methods use SGD momentum 0.9 and weight decay 0.0005 by default. The optimizer state policy defaults to **persist** across successive local experiences and after synchronization; it is still an explicit controlled setting where experimental ablations require reset behavior. Persisting state across a global download is a deliberate methodological choice, not exactly equivalent to freshly reinitialized local momentum.

## Exactly-64 training batches without discarding examples

The training sampler first permutes the authorized CURRENT-TRAIN examples each epoch, covers every index, and completes the final incomplete batch to size 64 by seeded replacement draws **from the same current authorized training pool**. No example is silently discarded with drop-last; no validation or test image enters that replacement pool. Repeated padding contributes to the training gradient, so the final stochastic full batch is **not numerically identical** to an ordinary smaller last batch. Its replacement draws are uniform over the current pool (and unbiased in expectation relative to a uniform sampling policy). All real optimizer forwards and backward passes use 64 inputs.

## Plain Experience Replay when history is small

When there is no historical memory, a client trains on physical batches of 64 current samples.

Once replay evidence is available, a single physical batch of 64 is assembled with **current plus historical TRAIN examples**, and **one** class-incremental supervised cross-entropy is evaluated on that combined batch. No second independently averaged historical loss is added. With `replay_ratio=1.0`, the allocation is 32 current and 32 historical images. Current examples are covered in epoch order; a current 64-image load yields two full mixed optimizer batches. For a different ratio `r>0`, the current portion is approximately `64/(1+r)`, rounded to an integer in [1,63]; the remainder comes from history.

If fewer historical images exist than are required, the historical portion samples **with replacement**, so even a memory containing 1–10 images never triggers a 1–10-image network forward. The last incomplete current portion is completed by draws from the same authorized current batch. No replay examples are invented or taken from future classes. This is a class-incremental **joint mixed-batch Replay variant**, not an unannounced doubling of the supervised gradient scale. The extra optimizer steps relative to current-only SGD are an intended consequence of consuming replay at the declared mix ratio.

## Common independent per-class reservoir policy

Both the pilot and the factorial Replay default use independent, persistent **Algorithm R** per-class sampling, quota **50 examples per class**. The pilot uses a positive per-class fraction `1/9` against 450 eligible CIFAR-100 TRAIN images/class; the factorial's default uses an explicit per-class quota 50. The two paths use the same Algorithm R reservoir implementation. The factorial's Replay-memory ablation overrides that quota with **0, 10, 20, 50 or 100 samples/class**; unlike the old total 1,000-example cap, these are actual effective settings. The dead legacy total-cap Replay branch is removed.

The server stores the full deduplicated union of **uploaded** client reservoir images with class labels and provenance and broadcasts the union to selected connected clients, without globally truncating it to 50/class. Offline clients keep their private reservoir and model until reconnection. The pilot's nonoverlapping 100-class curriculum results in 5,000 distinct server exemplars after the final flush if every class contributes its full 50-sample reservoir. This deliberately transfers raw images and is not privacy-preserving FL.

## Other training/evaluation paths

- GEM/A-GEM recompute constraint gradients from one **physical batch of 64**, sampling with replacement if a historical group has fewer than 64 distinct examples. There is no gradient accumulation from smaller reference forwards.
- ER-ACE, DER and DER++ use full historical model batches of 64, sampling with replacement if needed. Their method-specific losses remain distinct from plain Replay.
- EWC/MAS sample a bounded set of *real* TRAIN examples to estimate parameter importance. Model forwards are grouped into 64-image tensors with duplicate padding as needed; only gradients corresponding to genuine selected examples contribute to the estimator. This preserves individual-example importance rather than incorrectly using a mean gradient.
- Privileged diagnostic model forwards similarly pad to 64, then exclude duplicate padding from gradient/feature statistics.
- Validation and official TEST evaluation use 64-image model forwards. A final incomplete evaluation batch is padded **solely for computation**; its duplicate predictions are removed *before* per-example cross-entropy, confusion-matrix, denominators and accuracy are calculated. The TEST data are never used to update optimizer weights or client selection.
- Production CIFAR-10 launch/generation/verification scripts are also fixed to batch size 64; old batch-512 archives are historical only. Synthetic numerical test fixtures may deliberately use other sizes to verify mathematical identities without representing production runs.

## Aggregation and scientific limitations

The two pilot arms retain whole-model sample-weighted FedAvg (no custom classifier-row fusion). Returning offline client updates are still rebased with the explicitly noncanonical stale-delta adapter. Classifier rows for all client-known classes can update; genuinely future classes are frozen. Full server broadcast includes the accumulated class-indexed continual-learning artifacts. Every communication stage produces per-class server accuracy, with separate arrived-class and integrated-class analyses and held-out TEST-only audit.

Batch64, Replay blending, momentum persistence and changed reservoir budgets are scientifically meaningful protocol changes. The **v16 protocol and job IDs** prevent false comparison with previously executed v10–v15 checkpoints or results. The factorial ledger contains **32,133 jobs per backbone**, 99 local/server-method combinations and three fixed seeds; its v16 SHA digests are frozen in `configs/designs/stream_v16_batch64.json`.

## Validate and launch exactly two ResNet-18 jobs on GPU 0

```bash
git pull --ff-only origin main
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpu 0 --plan-only

mkdir -p logs
nohup bash scripts/run_primary_probe_cifar100.sh \
  --backbone cnn --gpu 0 --seed 0 \
  --replay-fraction 0.1111111111111111 --epochs 10 --lr 0.03 \
  > logs/resnet18_dual_batch64_v16.log 2>&1 < /dev/null &
```

No remote execution is claimed. Passing source-code unit tests and CI checks cannot establish the accuracy or GPU resource adequacy of the full experimental matrix.
