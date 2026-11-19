# v10: Strict task-ordinal concurrency on one GPU

## Contract applied throughout active repository

All CIFAR-100 jobs entering `cafcl.simulator.run_experiment` use `cafcl.parallel_clients.train_round_task_barriers`. This includes the two-arm primary Replay/FedAvg pilot, individual custom configurations, balanced jobs, and full factorial jobs for ViT-Tiny/16 or ResNet-18 (CIFAR stem). There is no old client-sequential training loop in the active simulator.

For each communication interval with two local tasks:

1. The simulator holds **five separate client model objects** (one per client) plus one server model object; each client has its own private weights, optimizer/replay/CL-state and, on CUDA, its own `torch.cuda.Stream`.
2. The five client Task 1 functions are submitted concurrently to `ThreadPoolExecutor(max_workers=5)`. All five join before any subsequent task starts.
3. The five Task 2 functions are submitted concurrently. Each client trains Task 2 using the client weights and continual-learning state produced by its Task 1. Again all five join.
4. Only after the Task 2 barrier does the simulator select four or five clients for federation. **Every offline client has still trained Task 1 and Task 2.**
5. Connected clients upload accumulated deltas **and their retained per-class 10% reservoirs**. The server merges their samples, updates global ViT via FedAvg and class-aware row fusion, and broadcasts the latest weights plus full shared CL archive. Offline clients get no broadcast.
6. This repeats for five training intervals covering 100 distinct classes; a terminal communication-only flush synchronizes remaining pending updates without fabricating tasks.

For the two-run preliminary campaign, a separate **baseline process** (5/5 participating) and **rotating-offline process** (4/5 participating) start concurrently, **both on one GPU**. They do not share model weights or sample archives across experiments. Logical model states: two servers + ten client models. The device executes CUDA work from ten independent client streams over two processes; actual individual kernel overlap depends on scheduling and resource availability. A single GPU does not guarantee literally simultaneous physical execution of all kernels; the enforced contract is concurrent execution requests, persistent independent model instances, and strict task barriers.

This is GPU memory intensive. The primary probe accepts effective optimizer batch sizes of **64** or **1024** through `--batch-size`; the physical pilot microbatch defaults to **4** to reduce resident activation cost. Microbatching does not alter the selected effective logical batch. OOM must cause an explicit run failure; there is no silent fallback to sequential training. Lower `--micro-batch-size` explicitly if necessary, and monitor GPU VRAM. ResNet BatchNorm running statistics reflect physical microbatches (as already documented).

## Reproducible launch on one-GPU mtechdeserver

```bash
ssh mtechdeserver
cd Correlation-Aware-FCL
git pull --ff-only origin main
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone vit --plan-only

mkdir -p logs
nohup bash scripts/run_primary_probe_cifar100.sh \
    --backbone cnn --gpu 0 --seed 0 --batch-size 64 --micro-batch-size 4 \
    --replay-fraction 0.11111111 --epochs 10 --lr 0.003 \
    > logs/primary_single_gpu_v10.log 2>&1 < /dev/null &
tail -f logs/primary_single_gpu_v10.log
```

The legacy `--gpus 0,0` spelling is supported; distinct GPU IDs like `--gpu 0` are explicitly rejected. Run outputs include the selected backbone and batch size in their isolated result directory, with two independent experiment workers. Every communication interval records `client_execution_mode`, `independent_live_client_models`, and `task_barriers` in `metrics.json`.

**Verification boundary:** CPU scheduling/barrier tests establish worker overlap and correct task ordering, not GPU hardware occupancy or accuracy. The single-GPU deployment must validate available VRAM and CUDA operation empirically. The algorithm is unchanged apart from execution ordering/physical model instances; A-GEM/GEM/replay and other CL methods use their existing objectives and synchronized histories.
