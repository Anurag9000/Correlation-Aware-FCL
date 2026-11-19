# v13 — The only active CIFAR-100 local CL/synchronization curriculum

## Single immutable training rule throughout all active CIFAR-100 jobs

The same protocol applies to **all 11 client learners** (SGD baseline, Replay, GEM, A-GEM, EWC, SI, MAS, LwF, ER-ACE, DER, DER++) and **all nine server strategies**, under every valid factorial configuration and both primary pilots. No second pathway with two tasks/client/round is permitted.

* **First training communication interval only:** each client sees two consecutive tasks, exactly two new labels per task. Client 0 gets (0,1), then (2,3); Client 1 gets (4,5), then (6,7), ... in the disjoint and naturally ordered five-client pilot. Each client uses its first task state while learning its second.
* **Server synchronization after Task 2:** select online clients, merge complete global model parameters and **all accepted local CL artifacts accumulated to date**, broadcast a model **and every method's available CL history** to every online client. Offline clients retain old model/history and privately accumulated contributions until reconnecting.
* **Every later training communication interval:** **exactly one two-label task per active client**, which is a continual-learning task **immediately** with respect to the full synchronized server history the client last received. No second task is inserted before server sync. All clients receive a task-ordinal barrier; the server does not merge until all assigned clients finish.
* **Five-client, 100-class disjoint case:** 20 classes in the first interval plus eight intervals of 10 classes = **nine** training intervals, 50 two-class tasks total, then one communication-only flush that receives all clients' pending updates. Model+CL version increments only on accepted upload aggregation.
* **Other client counts in factorial jobs:** same immutable 2 tasks/client in the first interval and 1 task/client/interval thereafter, with a partially populated final interval when class budget runs out. In all cases each configured local task contains exactly two distinct labels.
* Class recurrence, overlap, sample skew, selection, server optimizer, and CL method ablations retain their own semantic axes. Their **timing/task-count protocol** does not vary. Old \`task_length\` designs requesting 1 or 3 tasks/interval, or 4 classes/task, are **rejected from executable v13 jobs** rather than silently violating this rule.

## Exact causal state of a client after communication

Let \`G_t\` be the server's accepted global ViT+100-row classifier after communication t. The server also holds a cumulative state \`H_t\` including the deduplicated labeled exemplar archive and image bytes, trajectory logits for DER/DER++, client+original-experience provenance and GEM/A-GEM gradient-reference sample groups, the additive EWC/SI/MAS parameter importance precision and weighted-anchor numerators, the cumulative accepted class exposure ledger, and the **latest global LwF teacher** for version t.

For every **connected** client k:

\`\`\`text
client_model[k] := G_t  (all backbone parameters and classifier rows/biases)
client_received_history[k] := H_t  (all supported method artifacts, not just its current method)
client_active_method_statistics[k] := compatible components of H_t
\`\`\`

The next two-label input is trained immediately with respect to \`G_t\` and appropriate \`H_t\`. GEM/A-GEM recompute historical gradients on **their actual current model**, rather than using gradient tensors computed on another client's model version. EWC/SI/MAS reconstruct their accumulated quadratic regularizers from precision/weighted-anchor statistics. Replay/ER-ACE access the complete server history of shared images; DER/DER++ additionally match archived original logits when available. LwF uses the latest frozen global teacher. Plain SGD is retained as the **no-explicit-CL-mechanism baseline**; it starts from the current global weights but cannot be said to retain all past classes through a regularizer or memory loss.

The server **never** has access to an offline client's unsent new state, so \`H_t\` means all **accepted** uploads. Offline clients are not entitled to magically receive the current \`G_t,H_t\` until they reconnect; while offline, their next task continues from their last available model+CL snapshot plus their own private accumulated state.

**Important equivalence distinction:** the server accumulates sufficient CL evidence, **not all historical raw gradient vectors** (which are model-dependent and mathematically wrong to directly average). GEM/A-GEM retain history of labeled reference examples; EWC/SI/MAS fuse diagonal precision statistics. LwF uses a **latest rolling global teacher**, not an archive of every large teacher model ever synchronized. These are the appropriate method-specific sufficient statistics; no original paper requires every historical teacher checkpoint to be replayed.

## Synchronization integrity

\`FederatedCLState.assert_complete_download\` is invoked at **every selected-client broadcast** in the simulator, including no-update communication. It validates replay indices, raw image payload IDs, DER target IDs, GEM/A-GEM reference IDs, all three regularizer tables and their numeric tensors, class exposure, LwF teacher version/parameters, and globally integrated classes. Thus an incomplete download is a test/run error rather than silently being treated as successful synchronization.

The rotating-offline pilot uses the seeded five-client cyclic order, excluding one client in **each of the nine training intervals**; a complete communication-only closing flush synchronizes everyone. Unlike the former five-round protocol, offline clients can be excluded more than once; the rotation covers all five clients before any repeats.

## Existing study setting preserved

CIFAR-100 TRAIN/validation split, 32x32 native randomly initialized ViT-Tiny/16 and CIFAR-stem ResNet-18, 100-class shared linear+softmax classifier, current-pair restricted CE and frozen inactive classifier rows, local method-specific CL objectives, original FedAvg/FedAdam/FedAvgM server formulas, local concurrency barriers and two independent baseline processes sharing GPU 0, replay reservoir rules and class overlap scenarios remain as in v12. No past held-out test samples or future classes are permitted in local training.

## v13 scientific identity and validation

All new results use \`cifar100_native32_v13_first4_then2_\` or \`primary_cifar100_first4_then2_v13_\` roots. The simulator's \`PROTOCOL_VERSION\` is \`cifar100_native32_v13_first4_then2_global_cl\`. Old v12 checkpoints/results are not valid evidence for the new synchronization cadence.

The test module \`tests/test_first4_then2_v13.py\` checks exact 100-class, 5-client scheduling, cyclical offline cohort, full accumulation/broadcast of multiple CL method artifacts, and every CL method training **one pair immediately after first synchronization**. Legacy regression tests remain active. GPU VRAM feasibility and final CIFAR-100 accuracy must be established with actual runs; source/CPU tests cannot guarantee them.

## Launch two concurrent runs on the same one GPU

\`\`\`bash
git pull --ff-only origin main
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpu 0 --replay-fraction 0.10 --plan-only

mkdir -p logs
nohup bash scripts/run_primary_probe_cifar100.sh \
  --backbone vit --gpu 0 --seed 0 --replay-fraction 0.10 \
  --epochs 10 --lr 0.003 --micro-batch-size 4 \
  > logs/primary_v13_first4_then2.log 2>&1 < /dev/null &
\`\`\`

The \`0.10\` value gives **45 TRAIN images/class** in the repository's 450-example/class split. To require **50 retained images/class** (the previously requested 10% of the original unsplit CIFAR-100 500), use \`--replay-fraction 0.11111111\` explicitly. This retains the split and does not touch validation data.
