# V25 class-incremental Replay/FedAvg run record

This is the run-specific record for the completed V25 seed-0 pair. It preserves
the settings that can be established from the surviving protocol and status
notes after the server result directory was deleted. V25 predates the Task-IL
head migration: it used ordinary Experience Replay (ER) with a shared
class-indexed classifier. It did **not** use DER or DER++.

## Run identity and outcome

- Dataset: CIFAR-100, native 32x32 RGB images.
- Backbone: CIFAR-stem ResNet-18, randomly initialized.
- Continual-learning method: ordinary Replay (ER), combined with server-side
  FedAvg. No stored-logit distillation objective was enabled.
- Protocol: class-incremental, using a single shared 100-class output layer;
  no task-ID head selection at inference.
- Seed: 0.
- Arms: full participation and random-offline participation.
- Final test / validation accuracy:
  - Full participation: **46.78% test / 46.08% validation**.
  - Random-offline: **42.13% test / 42.12% validation**.
- Both arms integrated all 100 classes after 20 learning syncs and a terminal
  communication-only flush. This is a one-seed empirical result.

## Dataset and class stream

- The protocol uses the official CIFAR-100 training set. Fifty examples per
  class are held out for validation, leaving 450 training examples per class.
  The official CIFAR-100 test split is used for test evaluation.
- Five clients each receive one two-class task at every learning sync, for 20
  learning syncs total. The task pairs form an adjacent chain and overlap
  across clients and sync boundaries.
- Sync 1 assigns `(0,1), (1,2), (2,3), (3,4), (4,5)` to clients 0 through 4.
  The chain advances by one class at each later sync. Sync 20 assigns
  `(95,96), (96,97), (97,98), (98,99), (98,99)`.
- Each occurrence gets a disjoint partition of that class's 450 training
  examples. Over the complete stream, class 0 has one client/task owner, class
  98 has three, and every other class has two. All 100 classes are integrated
  after the final flush.

## Local model training

- Each arm has five client model states and one server model state.
- Each client trains for 10 epochs on its current two-class task.
- Incoming current-stream batches have size 64. The 10-epoch stream passes
  preserve the full current-task stream; a final short batch is padded from the
  same current-task training pool rather than dropped.
- Training augmentation is a four-pixel zero-padded random crop and random
  horizontal flip. It is applied on training reads, including replay reads.
- Optimizer: SGD with learning rate `0.03`, momentum `0.9`, and weight decay
  `0.0005`.
- Client momentum state persists across local tasks and communication/model
  synchronization; it is client-local and is not part of FedAvg model weights.

## Replay memory and sampling

- Each client keeps up to 50 private examples per class in an independent
  per-class Algorithm R reservoir.
- Each incoming 64-image current batch is divided into two current batches of
  32. Each produces one optimizer update with 32 current images plus 32 replay
  images: 1:1 replay by image count and two optimizer updates per incoming
  current batch.
- Replay sampling is class-uniform, then image-uniform within the selected
  class, over the client's currently available private examples and its latest
  synchronized server archive.
- Sampling is without replacement within a replay minibatch when the available
  archive has enough distinct examples; replacement is used for smaller
  archives.
- The server archive is a deduplicated union of accepted client exemplars, not
  a fixed global 50-per-class buffer. The expected final archive is 10,000
  examples: 50 for class 0, 150 for class 98, and 100 for every other class.
- An omitted client keeps its last received archive snapshot until it next
  synchronizes. It continues to use that snapshot together with its private
  examples.

## Participation, aggregation, and synchronization

- **Full arm:** all five client model uploads are accepted at every learning
  sync.
- **Random-offline arm:** all five clients train, but a seeded random selector
  omits one client's model upload at each learning sync; the server accepts the
  other four. This is random omission, not a guaranteed round-robin schedule.
- Only selected clients receive the committed server model and archive at a
  learning sync. An omitted client continues from its own last synchronized
  model and archive.
- The server uses sample-weighted whole-model FedAvg. It does not perform
  classwise classifier-row fusion, and it does not use a server optimizer such
  as FedAdam or FedAvgM.
- Because an omitted client keeps learning locally, its later upload may be
  stale. The repository's documented adapter rebases that client's local
  change against the current server state, scales it by `1/(1 + staleness)`,
  and caps a sole stale terminal-flush contribution at `1/5` of its already
  scaled delta. This is an asynchronous adaptation around FedAvg, not
  synchronous textbook FedAvg for the random-offline arm.
- After the 20 learning syncs, a communication-only terminal flush integrates
  pending updates; it performs no additional local training.

## Execution and evaluation

- The completed V25 run used a sequential wrapper: the full arm finished before
  the random-offline arm started. The arms therefore did not compete with each
  other for the GPU in this run.
- Each arm trained five client workers concurrently on GPU 0, with a server
  synchronization barrier after each learning sync. The specific GPU model,
  driver, and framework build are not preserved in the remaining V25 run
  record.
- Per-sync accuracy is class-incremental seen-class evaluation using the shared
  classifier, not Task-IL task-head scoring. At sync 10, full scored
  **42.47% test / 43.80% validation**. Random-offline scored
  **38.78% / 40.59%**; class 50 was still pending server integration, and the
  integrated-classes-only test score was **39.56%**.
- Final evaluation covered all 100 integrated classes after the terminal flush.

## Settings not recoverable from the surviving record

The raw V25 `config.json`, metrics, per-sync reports, and checkpoints were
deleted from Mtech during the requested result cleanup. The repository notes
preserve the protocol and scores above, but do not establish every low-level
implementation detail. In particular, the exact per-worker RNG derivation,
data-normalization constants, horizontal-flip probability, scheduler/gradient
clipping settings (if any), and exact sample-count weights used by the server
are not asserted here. The one-seed results do not establish variance or
statistical significance. The documented V25 result is class-incremental and
must not be presented as a Task-IL or DER result.

## Sources

- [V25 stream and Replay protocol](PRIMARY_PROBE.md)
- [V25 execution note and recorded metrics](STATUS.md)
- [Server update and stale-client adapter](SERVER_ALGORITHM_FIDELITY_V11.md)
- [Historical single-head and frozen-row training contract](CL_METHOD_FIDELITY_V12.md)
