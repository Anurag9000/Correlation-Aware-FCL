> **Historical protocol:** The sole current primary Replay/FedAvg recipe is in [PRIMARY_PROBE.md](PRIMARY_PROBE.md). This v8 protocol is retained as research history and is not a current FedAvg implementation or launch guide.

# Federated continual-learning synchronization — protocol v8

Every CIFAR-100 production job now uses **shared model and shared method-specific continual-learning state**. A client is **always** a continual learner: it trains all locally arriving two-class tasks in sequence, even if it is disconnected from the central server. Communication/selection only controls uploads and downloads, never whether local learning occurs.

## Transactional communication sequence

1. Every client sequentially trains each task with its own persistent local weights and CL evidence. The first and every subsequent local task use all compatible received history from its last server synchronization, plus all additional privately learned history.
2. Only participating clients upload their accumulated anchor-relative parameter changes and newly contributed method-specific artifacts. Nonparticipating clients retain all private learning, old server-archive snapshots, optimizer/method state, and pending changes without any server broadcast.
3. The server merges the accepted updates and newly contributed CL artifacts into a versioned, class-indexed global ledger. Training examples are deduplicated by canonical train-index provenance, and importance updates are additive only for **fresh** contributions. An uploader must not re-upload the previous server broadcast as new evidence.
4. The server commits its model and CL state, then sends the newly committed model **and the entire CL artifact snapshot** to all participating clients. Class identity and exemplar source-client/source-local-experience metadata remain stable; the client preserves its private information and combines it with the received archive.
5. A returning client uploads **all unsynchronized accumulated learning** and receives the entire latest global model and global CL state. The next *first* local task, e.g. (21,22) after earlier (1,2) → (3,4) and global classes 1–20, is trained continually with respect to those received artifacts. The second local task (23,24) adds that first task's additional private history, and so on.

## Artifact contract by local learning method

| Method | Server merge | First post-sync task |
|---|---|---|
| SGD | Global model and integrated class history; no fictitious memory or gradient history | Standard SGD from latest synced model |
| Replay | Provenance-deduplicated union of clients' **actual uint8 RGB** training examples and canonical labels; each client's **private** replay budget is 1,000 images, while server union is uncapped | Replay from entire received server archive + privately retained examples |
| GEM / A-GEM | Class-indexed historical exemplar/constraint supports, image bytes, original source client and local experience | Recompute GEM constraints or A-GEM reference gradients at the *current* model weights, using the entire eligible historical table |
| ER-ACE | Historical images and labels plus observed-class information | Continual replay / ER-ACE masking from all historical classes |
| DER / DER++ | Samples, labels, stored historical logits, first-contributor provenance | Distillation from all compatible historical teacher targets + replay |
| EWC | Fresh Fisher-diagonal precision and precision-weighted parameter anchors | Quadratic importance regularization from all synchronized classes |
| SI | Newly consolidated path importance and anchors (including importance from the last task before each communication) | Quadratic importance regularization from all synchronized classes |
| MAS | Fresh sensitivity-based diagonal importance and anchors | Quadratic importance regularization from all synchronized classes |
| LwF | Versioned committed global-model teacher, plus integrated class history | Distillation against server teacher from the **first** new local task |

**Gradient caveat:** GEM/A-GEM raw gradient vectors computed against different client models cannot be averaged as if they were gradients at the same parameters. The mathematically meaningful global table is the union of per-class historical constraint **evidence**; receiver clients recompute gradients at their current model. A raw-gradient aggregation would require a common parameter anchor and its own approximation controls. The implementation intentionally avoids false mathematical equivalence.

**Mixed methods:** The server now merges artifacts according to each uploader's actual local algorithm, including heterogeneous-client runs, not only the global default method. Every participating client receives an explicit *passive* snapshot of replay images, constraint references, DER logits, EWC/SI/MAS precision tables, teacher versions and origin metadata. Each client only applies the artifacts defined for its active local method. No incompatible gradient or teacher statistics are silently averaged.

## Invariants and validation

- No local history is erased on communication; reset-memory controls are disallowed for CIFAR-100 production.
- A client not selected for communication cannot receive any CL-state or model update.
- Offline private history is preserved until reconnection; only its acknowledged new state is added to the server ledger.
- No test or validation examples may enter the global exemplar archive.
- Provenance-safe GEM stage grouping does not confuse two clients' task IDs and does not crash on remotely sourced class histories.
- EWC's Fisher loss uses the current active-class objective for paired local tasks.
- SI accumulates and uploads importance at the end of **every** complete local task, including the second task in a communication interval.
- Every compatible learner consumes received state on the **first** local task after synchronization, verified by dedicated tests across replay, GEM, A-GEM, ER-ACE, DER, DER++, EWC, SI, MAS and LwF.
- All production CIFAR-100 jobs preserve native 32×32 resolution and effective batch size 1024; both ViT-Tiny/16 and CIFAR-stem ResNet-18 share the state protocol.

The version identifier is **`cifar100_native32_v8_full_cl`** and the primary two-arm Replay/FedAvg probe writes to `primary_cifar100_full_cl_v8_*`. All v7 checkpoint/status files are incompatible; do not silently resume v7 jobs under v8 semantics. The existing 38,370-per-backbone v6 geometric configuration ledger is unchanged, but its executions now use the new v8 semantics.

Run verification with:

```bash
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --plan-only
```

## Scientific limitations

This is explicit inter-client training-image sharing, not privacy-preserving FL. A complete class-indexed server archive can grow large and requires storage/transmission accounting. Diagonal EWC/SI/MAS aggregation and versioned LwF teacher distillation are research adaptations requiring evaluation. The server's metadata and shared training-index namespace are simulator conveniences, not a fully secure distributed transport. Passing offline tests is not evidence of good CIFAR-100 classification accuracy.
