# Historical class-indexed protocol (pre-Task-IL)

This document records the former single-head class-incremental design. New runs use task-specific heads and task-ID-assisted evaluation; see TASK_INCREMENTAL_PROTOCOL.md. The remainder is retained only to describe historical experiments.

# CIFAR-100 v3: sequential local learning and class-indexed synchronization

## Precise per-client clock

The *communication interval* is the outer loop. Within each interval there are `local_tasks_per_round` **sequential** local learning events per client. Each event has `classes_per_local_task` distinct labeled classes. Default is `(two-class task A) → (two-class task B) → (optional server communication)`. Every client runs these events even if it will be offline at the upcoming server aggregation.

This document describes the generic class-indexed pair protocol used by the factorial and compatibility runners. The current five-client primary experiment uses the adjacent-chain schedule documented in [PRIMARY_PROBE.md](PRIMARY_PROBE.md), not the geometry below.

For default **10 clients / 2 classes per task / 2 tasks per interval**, the first interval teaches classes 0–39, the second 40–79, and the third 80–99. No class is fabricated to fill the last interval. Clients that exhaust their scheduled events simply do not perform extra local updates. Training and communication are separate: a client may still upload previously accumulated updates even in an interval with no fresh training.

Local-method state (replay indices/logits, GEM/A-GEM memory, SI/MAS/EWC parameter importance, LwF teacher, etc.) is owned by the client and maintained across disconnections. **No client ever uploads raw replay samples to the server.** These clients continue updating their own model, and later upload accumulated *parameter deltas*.

## Headless global classifier

There is **one** classifier with an output row indexed by global CIFAR-100 class ID `0..99`. For shape-stable disconnected clients, all rows are allocated from initialization, but **unseen rows are inactive**. A task involving e.g. labels `(10,20)` computes supervised cross-entropy using *only* logits 10 and 20. Rows for all other classes are frozen exactly, including against momentum drift and CL regularizers. Shared feature-extractor weights may still change, so these restrictions do **not** mathematically guarantee absence of forgetting or classifier calibration bias.

A repeated label such as 10 in `(10,11)` and then `(10,20)` is still **the same global row 10** on that client. There is no duplicate task-specific classifier head or task ID at inference. In cross-client overlap two local copies may temporarily exist, but a single corresponding server row is merged by canonical class ID.

The global test evaluator masks **never-arrived** output rows; all arrived classes are tested, including classes held only by inactive clients. Thus never-integrated classes may lower global accuracy without constituting forgetting of previously trained classes.

## Offline versus online clients

At each interval `t`:

1. Generate client-local tasks and deliver fresh private training data to all clients.
2. On **each client's persistent private model**, learn tasks sequentially using that client's configured local method. Update method-specific historical state and pending class-update counts.
3. Select the online clients. With `participation=.8` and `num_clients=10`, precisely 8 clients are eligible to upload; 2 are offline. `offline_duration_rounds=D` creates seeded cohorts whose disconnections last D full intervals, overriding independent random selection as a deliberate ablation.
4. Each online client uploads its accumulated change since its **last received server model**. For a client with anchor `A_k` and private model `L_k`, the server-frame candidate is `G_t + (L_k-A_k)`. This is an explicitly stale-delta variation of federated aggregation, **not identical to classic synchronous FedAvg**.
5. Apply the chosen server update algorithm on participating uploads. For each classifier row `c`, consider only clients that have trained label `c` since their last synchronization. For basic average-style methods, fuse using recorded class-update counts and any historical row contribution. Other server update algorithms apply their own candidate corrections followed by row-wise class-aware retention. Increment class exposure counters once per actual local update contribution.
6. Broadcast the new server model to **online clients only**; their full shared model weights are replaced while their private CL state remains. Offline clients do not download or reset anything.
7. Record local/online/offline timelines, class-specific upload counts, server model version, class-wise accuracy, and unsent updates for audit.

**Example:** Client 4 is offline after learning `(8,9) → (28,29)` and remains offline after `(48,49) → (68,69)`. Upon rejoining after `(88,89)` it uploads accumulated effects from **all five** binary tasks (ten class identities) in one server communication. Client 6 analogously can accumulate `(12,13),(32,33),(52,53),(72,73),(92,93)` while offline. No server access to private raw images is required.

## Final communication-only flush

Because a finite 100-class disjoint stream ends after 3 *training* intervals at default capacity, clients randomly offline in the last interval might never reconnect. `stream_terminal_flush=True` adds one **explicitly labeled, no-training server communication step** that selects all clients to send pending updates. It is **not** counted as a four-new-class learning interval. Disable it to study the never-synchronized terminal server state.

## Scientific distinctions and limitations

- **Local sequential supervised tasks** do not make this conventional task-ID-assisted Task-IL. The server and inference use one global class-indexed classifier and no task ID; research classification is global class-incremental learning with known labels.
- Preallocated output rows are used for shape stability instead of physically expanding network tensors. Unseen rows are never trained locally or merged until activated.
- Frozen *classifier rows* do not freeze the shared representation; CL mechanisms still matter, and catastrophic forgetting remains possible.
- A class-update count is a weighted-fusion heuristic, not an optimal proof of unbiased aggregation. Old-class calibration and sample/effective-update weight mismatches need GPU study.
- Some local mechanisms (DER, EWC, MAS, ER-ACE, LwF) are faithful to the declared experiment code but are approximations of their literature counterparts. GEM's quadratic program currently uses SciPy on CPU; neural-network training can use CUDA.
- Selectors using simulated class holdings or gradient relationships are oracle research controls, not secure privacy-preserving selectors.
- No official CIFAR-100 training result or independent experimental validity claim is established by unit tests alone.
