# Federated continual-learning state synchronization (protocol v7)

## Central requirement and privacy
Selected clients upload model updates **and method-specific historical state**. The server merges those client-only contributions and broadcasts the committed global model **and consolidated continual-learning state** to all connected clients. Offline clients continue local learning without receiving anything. The client's old global-archive snapshot is replaced with the new combined archive when it reconnects. The existing full-sync ablation intentionally overrides normal disconnection semantics.

**Raw-image sharing is explicit in this research variant.** The server stores compressed 3x32x32 uint8 image payloads and canonical labels for accepted replay/GEM/DER exemplar IDs, in addition to their source IDs. Recipients receive those payloads. The simulator also uses a shared training-index namespace when constructing replay dataloaders. A physical FL implementation needs transport/consent/access-control and a receiver-local dataloader; this is not privacy-preserving federated learning.

## Local/private versus global capacity
The plain Replay algorithm retains at most **1,000 private exemplars per client** (config: replay_buffer_capacity). GEM, A-GEM, ER-ACE, DER and DER++ retain their previous method-specific per-class budget, while uploading and receiving their eligible historical exemplars. The server keeps a provenance-deduplicated union **without a global 1,000-example cap**. Therefore a client's effective set of shared + private replay examples can be larger than 1,000. The synchronized archive is never accidentally clipped by the private-client budget.

Historical image samples are keyed by canonical CIFAR-100 label and training-index provenance; a rebroadcast sample is not re-uploaded as new evidence. The server stores actual compact image bytes, not only class names. All data are TRAIN examples; validation and test samples are prohibited.

## Method-specific merge and rejoin behavior

| Local algorithm | Centralized state and merge | Synchronized client uses |
|---|---|---|
| SGD | No replay/importance state; integrated-class history is shared | Updated global model |
| Replay | Full deduplicated union of private retained image samples | The full federated replay archive plus its own private buffer |
| GEM | Class-indexed exemplar/constraint-support table, with full image payloads | Recomputes constraints using its current full model gradient |
| A-GEM | Same shared historical support table | Samples reference examples and recalculates the current-model A-GEM reference gradient |
| ER-ACE | Replay exemplars and global class-history state | Shared historical examples and class-aware loss masks |
| DER/DER++ | Exemplars plus canonical per-sample historical logits, first-source provenance | Shared stored logits/labels and historical examples |
| EWC | Only **fresh** diagonal Fisher/importance contributions, additive precision and precision-weighted parameter anchors | Consolidated Fisher and reference weights |
| SI | Only **new** locally consolidated path-importance increments, additive weighted references | Consolidated omega and anchors; local optimization path remains private |
| MAS | Fresh output-sensitivity importance and weighted anchors | Global consolidated omega and anchors |
| LwF | Versioned, recently committed global-model teacher, not a naive average of stale client teachers | Distillation from the server's teacher on the first post-sync experience |

There is **no mathematically legitimate way to average GEM/A-GEM raw gradient vectors from different local model parameter frames** and treat them as equivalent historical constraints. The server therefore synchronizes the complete per-class *constraint evidence* and receiver models compute their own current reference gradients. This retains the intended GEM/A-GEM mathematics; no stale gradients are silently reused.

Historical regularizers are communicated as nonduplicated *incremental natural parameters* rather than echoing the last broadcast Fisher/omega back into the sum. The teacher is versioned by synchronized global model. DER replay targets retain the first uploaded source for any repeated training-index key.

## Exact communication sequence
1. Each client sequentially learns every new two-class task on its own locally persistent model, even when disconnected.
2. The client updates its private replay/history/importance/teacher state.
3. Only online clients upload their accumulated, anchor-relative model changes and pending method-specific CL state.
4. The server validates and merges accepted model changes and **new, client-owned** state into the central versioned tables.
5. The server commits the aggregated class-indexed model and CL tables and broadcasts both to **online** clients, resetting their acknowledged pending deltas.
6. Offline clients keep their private weights, older shared archive and unsent increments, with no new server information.
7. When they reconnect, their pending private history is merged and they receive the now-current global model and entire shared archive.
8. A terminal communication-only flush can integrate the final unsent clients.

## Reproducibility, experiments, and limits
The model/simulator checkpoints include the server CL state and every client's private/received state. Round metrics include server archive sizes, image-byte counts, method-state versions, and private/received memory sizes. The protocol ID is **cifar100_native32_v7_shared_cl**, and output roots contain **v7_shared_cl**. Old v6 model-only statuses and model checkpoints are incompatible with v7 execution.

The existing 38,370-job-per-backbone v6 *geometric* registry is retained as a finite design. The v7 protocol changes what each run means; v6 results are NOT directly comparable. The default primary two-job CIFAR-100 pilot has five clients, five rounds, one different client offline per round, Replay + FedAvg, seed 0, native 32x32 ViT-Tiny/16, effective batch 1024, and a 1,000-example private buffer. Its two arms run through the existing concurrent GPU launcher.

Important limitations: the server and all clients share a simulation-level training-index namespace, rather than distinct physically hosted datasets; cross-client image sharing changes the privacy and communication assumptions; diagonal regularizer merging and versioned teacher fusion are approximations that need independent experimental validation; GPU efficacy is not yet established.

### Receiver-side replay data path
On synchronized clients, the continual-learning dataloader now uses `ClientArchiveDataset` for all shared indices, reconstructing normalized 32×32 training tensors from the actual uint8 exemplar payloads. Newly arrived client-owned examples continue to come from the local training dataset. This prevents synchronized replay from silently depending on privileged raw data from other clients, although the simulator still uses global integer IDs for provenance and routing.
