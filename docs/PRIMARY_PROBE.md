# CIFAR-100 Replay/FedAvg paired schedule

This document records the V25 stream recipe. New configurations use Task-IL task-specific heads as specified in TASK_INCREMENTAL_PROTOCOL.md; this schedule alone does not define the classifier or evaluation protocol. V25 result tables below predate that migration and are historical class-IL measurements.

## Fixed recipe

| Item | Current setting |
|---|---|
| Dataset | Official CIFAR-100 at native 32×32; 450 training examples/class |
| Model | CIFAR-stem ResNet-18, random initialization, shared features with task-specific heads |
| Training views | Four-pixel zero-padded random crop and horizontal flip on every train read, including replay reads |
| Local learner | Replay; 50 examples/class per client, Algorithm R reservoir |
| Replay update | Each incoming physical batch is split into two 32-image current batches; each is paired with 32 replay images sampled by choosing a currently visible class uniformly, then an image uniformly within that class |
| Replay read size | 32 replay images per 32 current images at each update (1:1) |
| Server | Sample-weighted whole-model FedAvg; no classwise head fusion |
| Local optimizer | SGD, lr 0.03, momentum 0.9, weight decay 0.0005; momentum persists across local tasks and communication |
| Local training | Batch 64; 10 epochs per two-class task |
| Clients | Five independent client models per experiment process |
| Stream | One two-class task/client at every learning sync, beginning at sync 1; 20 learning syncs |
| Class geometry | Adjacent pairs overlap across clients and across sync boundaries |
| Participation | Full 5/5 versus four model uploads selected randomly from five each learning sync |
| Offline learning | All five clients train; an omitted client keeps its last received model and replay archive until selected again |
| Concurrent execution | Two independent arm processes on GPU 0; five CUDA-stream client workers/process; ten client models train concurrently, plus one server model/process |
| Closing step | One communication-only terminal flush after the 20 learning syncs |
| Seed | 0 |
| Evaluation | Task-ID-assisted Task-IL score; overlap probabilities are averaged by global class and renormalized |

## Exact class schedule and memory counts

Sync 1 assigns `(0,1),(1,2),(2,3),(3,4),(4,5)` to clients 0–4. Each following sync continues the chain, carrying the last class of one pair into the next pair. Sync 20 assigns `(95,96),(96,97),(97,98),(98,99),(98,99)`. The repeated final pair keeps every client training at the final sync while giving class 99 two owners. The resulting class ownership counts are: class 0 once, class 98 three times, and every other class twice.

Each occurrence receives a disjoint partition of that class's 450 raw training images. Each client stores up to 50 of its own examples per class. After integration and the final flush, the expected deduplicated server archive contains 10,000 exemplars: 50 for class 0, 150 for class 98, and 100 for every other class. The aggregate archive is broadcast only with a selected client's model synchronization. A client omitted from upload keeps its prior archive snapshot until it is selected again.

## Replay semantics

Each incoming 64-image physical batch produces two optimizer updates. Each update contains 32 current images and 32 Replay images, so the replay count equals the current-image count for every incoming batch. For every replay draw, the learner first chooses uniformly among classes represented in the client's currently available private and synchronized archive, then chooses uniformly among that class's stored images. Replay images are sampled without replacement within each minibatch whenever the available archive has enough distinct images; smaller buffers use replacement. Per-client, per-sync reports include replay draw counts by class for audit. An offline client therefore samples from the archive snapshot it last received, together with its private examples. The current stream is traversed each epoch, with the final physical batch padded from the same training pool rather than dropping examples. The terminal communication-only flush performs no local training.

The adapted FedCLASS Replay path retains its separate source-specific full-buffer rule while using its original first-50-per-class selector. GEM/A-GEM evaluate the complete retained buffer for their projection constraint at each incoming update. The separate GLFC/LGA/FEAT source-method runners use their method-specific source loaders and selectors; their execution details are documented in [NATIVE_OFFLINE_FIDELITY.md](NATIVE_OFFLINE_FIDELITY.md). No-replay modes do not read a replay archive.

## Execution topology and offline semantics

Each arm owns five independent client models and one server model. At every learning sync, the five client tasks are submitted in one five-worker wave. Client model updates finish at the round barrier before the server aggregates uploads. The full arm accepts all five uploads. The random-offline arm chooses one client via the seeded random participation selector and accepts the other four uploads. This is a fresh random draw each sync, not a round-robin guarantee; a client may be omitted more than once or not at all. All clients keep learning from their own last synchronized model and last received archive. Only selected clients receive the committed model weights and archive at a sync.

The two arm processes start concurrently and use physical GPU 0. This specifies concurrent independent workers and CUDA streams; a single GPU scheduler may interleave kernel execution and does not guarantee ten kernels execute at the exact same instant. The server model updates only at the synchronization barrier.

## Run

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
bash scripts/run_primary_probe_cifar100.sh --sync-every-batch --plan-only
```

The runner writes a Task-IL V27-specific result root so these settings cannot resume a checkpoint from the prior all-known-head replay rule. A plan-only run validates the schedule and reports the seeded per-sync outage plan without launching training. The currently running one-seed V26 Task-IL pair predates source-task replay routing; prior V25 results in this document used the old single-head classifier.

The separate `--sync-every-batch` option prepares V28 with one server round trip after each active client's physical batch. With ten local epochs and this task schedule, the plan-only audit expands 20 logical task rounds into 1,630 learning syncs, followed by the terminal flush. The final short batch is padded from earlier examples in the same client's current task and those padded examples participate in loss, matching the fixed 64-image batch rule. This option has only passed plan-only validation; it has not been trained or scored. Because it changes communication frequency and the number of optimizer aggregation barriers substantially, compare it as a separate protocol variant, not as a same-sync-count rerun.

## Historical result from the prior protocol

The previous completed paired run scored 38.58% with full participation and 30.58% with rotating offline participation. It used nine learning syncs, a disjoint class schedule, and sampled-minibatch replay. Its one-seed result does not validate the current adjacent-chain schedule, random outage policy, or class-uniform 1:1 sampled Replay rule. Its reported gains over the preceding matched V19 run were 17.92 points for full participation and 15.86 points for rotating offline; these comparisons are also historical.

The earlier rotating-offline arm used the marked stale-update adapter because disconnected clients continued local learning from older model states. It was not synchronous textbook FedAvg. The current random-offline arm retains that delayed-update condition, so this remains an explicitly asynchronous protocol adaptation.

## Interpretation limits

This is a one-seed experiment. Class ownership and stored exemplar counts do not imply equal class accuracy. The outage plan is random with a fixed seed and does not guarantee every client is omitted. Empirical performance and GPU memory use must be measured from the completed run; the plan itself provides neither.
