# Canonical class-indexed local continual learning and federated synchronization (v2)

## What the code now does

Every client learns on its private model while connected or disconnected. At a communication round, participating clients upload accumulated updates, and **only synchronized clients** replace their private model weights with the newly aggregated global weights. Replay buffers, GEM/A-GEM memory, EWC/SI/MAS statistics and other private CL state survive ordinary synchronization. Offline clients retain their weights, unsent updates and memories. `full_sync=True` is an explicitly different synchronization ablation.

The output layer has **one row per canonical class ID**, identical across clients and the server. There are **no task-specific classifier heads**. Class 10 on any client always maps to output row 10 at the server, including overlapping client streams and recurrent encounters. **Important:** the present implementation preallocates rows for known CIFAR labels; it activates them logically, but it does not physically grow the tensor at discovery time.

## Two-class local training

When two class labels `a,b` arrive at the same client and logical round, the local CE objective uses **only logits a and b**. A class sequence `(0,1)`, then `(2,3)` trains the first binary episode before the second, on the same evolving private model, before any server update.

The classifier rows other than the active two are frozen, including after GEM/A-GEM projections, local regularization and retained SGD momentum. The feature extractor can still change: frozen class rows do **not** guarantee frozen predictions. GEM, A-GEM, replay, EWC, SI, ER-ACE, DER, DER++, MAS and LwF retain their method-specific local objectives under the active-pair rule. Historical memory or distillation still regularizes the backbone.

An old label reused in a new pair, e.g. `(10,11)` followed by `(10,20)`, uses the **same canonical row 10**, never a second head. Two clients learning class 10 both map to that row upon upload.

### Compatibility limitation — important

`RunConfig.class_pair_protocol='strict'` rejects singleton arrivals rather than silently treating a one-class batch as a meaningful binary classification task. The original enormous job ledgers use `class_pair_protocol='compatible'` for backward compatibility: simultaneous labels are trained in pairs when possible, but **unpairable singletons still use a legacy full-softmax objective**. Those jobs are **not fully compliant** with the requested two binary episodes per client per step. Inspect `local_training_events[].classes` before treating a run as conformant.

This is a hard mathematical constraint for some original job layouts. Four class-disjoint CIFAR-10 clients each seeing four distinct class labels simultaneously require **16 labels**, while CIFAR-10 supplies only ten. A valid strict layout requires fewer data-bearing clients, different time schedules, permitted class overlap, or a richer dataset such as CIFAR-100. The historic 67,938/69,722 jobs are not certified strict solely by this update.

## Class-wise server merge

The client's uploaded weights are rebased using `G + (L - A)`, where `G` is the current global model, `L` is its private model, and `A` is the last global model it actually downloaded. This preserves offline learning and is a **stale-delta adaptation**, not ordinary synchronous FedAvg when the client has missed rounds.

For every global classifier row c, the server reads only contributions from clients that **trained that class since their last synchronization**. With `classwise_head_merge=True`, prior server update counts and fresh client-local class update steps weight a class-specific fusion. Unchanged classes keep their previous row. For non-FedAvg optimizers, the algorithm-specific candidate delta is damped according to the share of new class updates. All counts are saved in resumable full-state checkpoints.

This is a **proposed class-wise fusion algorithm**, not a stock FedAvg/FedAdam implementation. Use `classwise_head_merge=False` if an uncontaminated original-server-method control is required. The network backbone still follows the selected server optimizer or aggregation method. Class-specific row fusion is a distinct research choice that must be ablated.

On every aggregation success, **the server model replaces each selected client's shared weights exactly**, while preserving its historical CL memory; nonselected client weights are never overwritten. The next time an offline client participates, it uploads its accumulated effect before the replacement.

## What is verified, and what is not

Offline tests cover all eleven local methods on two successive two-label episodes within the **same logical round**, disabled-client learning and later reconnection, remembered class histories, global row identity, frozen inactive classifier rows, checkpoint protocol gating and stale update accumulation. The suite ran **96 offline tests** at the time of this engineering update.

The current model does **not** dynamically resize classifier tensors; it preallocates a canonical class registry. Nor are all historical class-holder allocations, arrival schedules, partial participation patterns and overlap variants converted to **strictly valid** two-pair-per-client jobs. Strict full-ledger feasibility checking, regenerated scientific factorial plans and real CIFAR-10/CIFAR-100 GPU results remain open work. No claim of exhaustive research closure or official CIFAR execution is made.

**Terminology:** sequential local two-class episodes without task-specific inference heads constitute **task-structured local continual training with class-incremental global inference**, not conventional task-ID-assisted task-IL inference.
