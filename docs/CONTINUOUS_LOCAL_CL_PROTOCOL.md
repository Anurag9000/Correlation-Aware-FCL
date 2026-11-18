# Continuous client-local continual learning, independent of server participation

**Training protocol:** `continuous_local_cl_v1`

This protocol applies to **every** registered CIFAR-10 and CIFAR-100 job, across all 11 local learning methods and all 9 server methods. The original and factorial experiment ID counts remain unchanged; **the semantics of those jobs change**. No new CIFAR GPU benchmark results are claimed.

## One logical communication round

1. Deliver the due class-data events to **all** clients, independent of selection. Each client is a private, persistent continual learner, processing new class events **one at a time in order**. A later class label is not visible to its local learner before the corresponding event.
2. Every client with local training data performs local training **even when it will not participate in federation**. Clients continue on their own previous local model rather than downloading the current server model. When no new event arrives, they continue on their accessible active data under the configured pending-data policy.
3. The selection policy chooses the clients permitted to **communicate**, not the clients permitted to train. Some selectors use simulator-privileged data and must not be marketed as privacy-preserving.
4. Selected clients upload their **accumulated private model change since their last download**, with total local training counts, accumulated class identifiers and model-version metadata. No hidden client's accumulated updates are discarded.
5. The server executes the requested aggregation method (FedAvg, FedAvgM, FedAdam, uniform, class-balanced, conflict-aware, projection, inverse-participation or retention-aware).
6. Once an update is accepted, selected clients receive the committed global model. Their private CL state (episodic memory, importance weights, historical teacher, prior experience trace and optional optimizer state) survives this download; only acknowledged pending updates are cleared. Inactive clients do **not** upload or download. The sole exception is the explicitly configured **`full_sync=True` experimental ablation**, which intentionally tests eager broadcasts to inactive clients.
7. Global evaluation still includes **all arrived classes**, even when held only by never-selected clients. The server must separately report never-integrated classes and actual forgetting. Full-state round checkpoints preserve all private client state.

This corrects the previous implementation, which trained only selected clients and overwrote their private model with the current global model **before** local learning.

## What an offline client sends when it reconnects

Let `G_t` be the current server model, `A_k` the server model client `k` last downloaded, and `L_{k,t}` the client's current privately trained model (including classes learned while offline).

Its entire pending local update is

`Delta_{k,t} = L_{k,t} - A_k`.

The server-frame client candidate is

`W_{k,t} = G_t + Delta_{k,t}`.

A FedAvg server computes `G_(t+1) = sum_k n_k W_(k,t) / sum_k n_k`, where `n_k` is the count of examples used by that client in locally accumulated training since its last acknowledged upload.

When `A_k = G_t`, the formula reduces **exactly to ordinary synchronous FedAvg**. When `A_k` is stale, it is an **explicit asynchronous stale-anchor delta adaptation**, not ordinary synchronous FedAvg. It preserves missed updates but introduces possible stale-gradient interference. Server optimizers are fed these server-frame weights consistently. Per-client anchor versions and unsent step counts are logged.

Example: client 3 last downloaded server version 0, privately learned classes 5, 6 and subsequently 19, 20, while excluded from communication. Its `L_3-A_3` includes the entire local trajectory. On reconnection this contribution is aggregated **before** its private model is replaced by the acknowledged server model.

## All supported local continual learners

| Local method | State retained across inactive rounds | New-class training |
|---|---|---|
| SGD | Private model parameters and optional optimizer momentum; **no explicit anti-forgetting memory** | Ordinary gradient updates on new class |
| GEM | Class/stage memory and projection statistics | Project gradient to satisfy constraints on historical classes |
| A-GEM | Bounded historical memory and reference-gradient logic | Project conflicting gradient against past-memory gradient |
| Replay | Bounded private sample buffer | Mix historical samples with newly arrived class |
| EWC | Diagonal Fisher surrogate and parameter anchor | Penalize moving important historical parameters |
| SI | Parameter path-integral importance, omega, anchor | Protect historically important parameters |
| ER-ACE-style | Private replay examples and seen-label metadata | Asymmetric stream masking and replay |
| DER | Private historical examples and stored logits | Replay-logit consistency |
| DER++ | Historical examples, logits and labels | Replay-logit consistency plus supervised replay |
| MAS | Parameter importance and historical anchor | Importance regularization |
| LwF | Historical local teacher | Distillation while learning new examples |

All implementations retain their existing paper-fidelity limitations; continuity does not convert research adaptations into exact reference-method reproductions. EWC/MAS consolidation follows a local experience event or stage boundary, while SI/LwF transitions use **per-client local experience indices**. FedProx anchors to the **last downloaded** server model, not to a server state an offline client could not observe.

**Private memory is not uploaded as raw images or buffers.** In conventional federated learning, it remains on the client; its effect is carried by model changes. A bounded buffer is not a promise to retain every original sample indefinitely.

## Class-disjoint feasibility

If every one of K clients must learn at least two **different** classes and class ownership is disjoint, at least `2K` unique classes are necessary. CIFAR-10 has ten labels: ten class-disjoint clients **cannot all** have two different-class experiences. The primary A-GEM pilot correctly pairs the ten labels across **five** clients, with one earlier and one later class per client. Some original 10-client disjoint CIFAR-10 jobs have only a single class per data-holding client and should **not** be interpreted as proving per-client class-incremental forgetting. Check `local_training_events` in the checkpoints before making method-level claims.

## Versioned outputs and tests

`src/cafcl/simulator.py` embeds `PROTOCOL_VERSION='continuous_local_cl_v1'` into status files, round checkpoints and saved models. A previous-protocol completed job or resumable checkpoint is **rejected**, not silently reused under the same job ID.

Every round's `metrics.json` includes `locally_trained_clients`, `uploaded_accumulated_classes`, `uploaded_since_version`, `uploaded_accumulated_steps`, `client_unsent_steps`, server `model_version` and client-local training records.

Run `PYTHONPATH=src pytest -q` (79 tests in the local offline run), and specifically `PYTHONPATH=src pytest -q tests/test_continuous_local.py` (15 tests). These test every local learner offline, preservation of its history while disconnected, delayed reconnection, class-by-class sequential learning within the same logical round, stale-anchor arithmetic and checkpoint/resume identity.

This is a **synthetic/offline regression validation**. It is not a successful CIFAR-10/CIFAR-100 GPU training result.

## Important qualifications

- Time uses synchronous logical rounds; network delays, asynchronous wall-clock execution and distributed device concurrency are not physically simulated.
- Every data-bearing client trains once per logical round on its current event/buffer; multiple class events in one round are trained sequentially. This is not the same as giving each class equal training time.
- A selected client without any local training since its last download does not fabricate an upload. Under the explicit `full_sync` ablation, inactive clients do receive updates and that condition differs from communication-free nonparticipation.
- Existing factorial counts and batch-size-512 configurations remain; computational work will increase substantially because unselected clients now train too.
- The test split remains in the current online evaluator and some history-aware selectors use that history. Before claiming leakage-free adaptive selection or publication conclusions, separate train-derived validation feedback from the official test set.
