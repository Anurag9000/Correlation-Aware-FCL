# Current CIFAR-100 Replay/FedAvg paired protocol

This document specifies the active V23 primary experiment. It supersedes the older nine-sync disjoint-class/rotating-offline launch instructions. Historical result tables below remain labeled as results from that older protocol.

## Fixed recipe

| Item | Current setting |
|---|---|
| Dataset | Official CIFAR-100 at native 32×32; 450 training examples/class |
| Model | CIFAR-stem ResNet-18, random initialization |
| Training views | Four-pixel zero-padded random crop and horizontal flip on every train read, including replay reads |
| Local learner | Replay; 50 examples/class per client, Algorithm R reservoir |
| Replay update | Each incoming physical batch plus every currently available unique private and synchronized archive example contributes gradients before one optimizer step |
| Replay read size | Batch-64 chunks; the full buffer is traversed at each incoming update |
| Server | Sample-weighted whole-model FedAvg; no classwise head fusion |
| Local optimizer | SGD, lr 0.03, momentum 0.9, weight decay 0.0005; momentum persists across local tasks and communication |
| Local training | Batch 64; 10 epochs per two-class task |
| Clients | Five independent client models per experiment process |
| Stream | One two-class task/client at every learning sync, beginning at sync 1; 20 learning syncs |
| Class geometry | Adjacent pairs overlap across clients and across sync boundaries |
| Participation | Full 5/5 versus four model uploads selected randomly from five each learning sync |
| Offline learning | All five clients train; an offline client's model stays unsynchronized while it receives the full replay archive |
| Concurrent execution | Two independent arm processes on GPU 0; five CUDA-stream client workers/process; ten client models train concurrently, plus one server model/process |
| Closing step | One communication-only terminal flush after the 20 learning syncs |
| Seed | 0 |
| Evaluation | All 100 classes after the terminal flush |

## Exact class schedule and memory counts

Sync 1 assigns `(0,1),(1,2),(2,3),(3,4),(4,5)` to clients 0–4. Each following sync continues the chain, carrying the last class of one pair into the next pair. Sync 20 assigns `(95,96),(96,97),(97,98),(98,99),(98,99)`. The repeated final pair keeps every client training at the final sync while giving class 99 two owners. The resulting class ownership counts are: class 0 once, class 98 three times, and every other class twice.

Each occurrence receives a disjoint partition of that class's 450 raw training images. Each client stores up to 50 of its own examples per class. After integration and the final flush, the expected deduplicated server archive contains 10,000 exemplars: 50 for class 0, 150 for class 98, and 100 for every other class. The full aggregate archive is broadcast to every client each sync, including the client excluded from model upload.

## Replay semantics

For each incoming batch update, Replay accumulates the incoming batch's mean supervised gradient and the mean gradient across the entire currently available replay archive. The fixed replay ratio of 1 weights the two means equally. The archive is traversed in chunks of at most 64 examples; it is not sampled down to one replay minibatch. Current incoming training indices are traversed each epoch, with the final physical batch padded from the same training pool rather than dropping examples. The terminal communication-only flush performs no local training.

The adapted FedCLASS Replay path follows the same full-buffer-per-incoming-update rule while using its original first-50-per-class selector. GEM/A-GEM evaluate the complete retained buffer for their projection constraint at each incoming update. The separate GLFC/LGA/FEAT source-method runners use their method-specific source loaders and selectors; their execution details are documented in [NATIVE_OFFLINE_FIDELITY.md](NATIVE_OFFLINE_FIDELITY.md). No-replay modes do not read a replay archive.

## Execution topology and offline semantics

Each arm owns five independent client models and one server model. At every learning sync, the five client tasks are submitted in one five-worker wave. Client model updates finish at the round barrier before the server aggregates uploads. The full arm accepts all five uploads. The random-offline arm chooses one client via the seeded random participation selector and accepts the other four uploads. This is a fresh random draw each sync, not a round-robin guarantee; a client may be omitted more than once or not at all. All clients keep learning from their own last synchronized model. At each sync, every client receives the complete committed replay archive, but only selected clients receive the committed model weights.

The two arm processes start concurrently and use physical GPU 0. This specifies concurrent independent workers and CUDA streams; a single GPU scheduler may interleave kernel execution and does not guarantee ten kernels execute at the exact same instant. The server model updates only at the synchronization barrier.

## Run

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The runner writes a V23-specific result root so these settings cannot resume a checkpoint from a prior protocol. A plan-only run validates the schedule and reports the seeded per-sync outage plan without launching training. The current protocol has not acquired empirical results until both worker processes finish successfully.

## Historical result from the prior protocol

The previous completed paired run scored 38.58% with full participation and 30.58% with rotating offline participation. It used nine learning syncs, a disjoint class schedule, and sampled-minibatch replay. Its one-seed result does not validate the current adjacent-chain schedule, random outage policy, or exhaustive replay rule. Its reported gains over the preceding matched V19 run were 17.92 points for full participation and 15.86 points for rotating offline; these comparisons are also historical.

The earlier rotating-offline arm used the marked stale-update adapter because disconnected clients continued local learning from older model states. It was not synchronous textbook FedAvg. The current random-offline arm retains that delayed-update condition, so this remains an explicitly asynchronous protocol adaptation.

## Interpretation limits

This is a one-seed experiment. Class ownership and stored exemplar counts do not imply equal class accuracy. The outage plan is random with a fixed seed and does not guarantee every client is omitted. Empirical performance and GPU memory use must be measured from the completed run; the plan itself provides neither.
