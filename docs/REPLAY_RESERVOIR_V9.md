# CIFAR-100 two-run pilot: per-class 10% reservoir replay (v9)

Both concurrent arms use Replay + FedAvg and retain the v8 full server-archive synchronization mechanism. The difference is the **private sample selection rule**. Each client's independent reservoir uses the classical online Algorithm R with a fixed per-class capacity, and never receives raw hidden-client data until that client connects.

## Exact class quota and algorithm

Each CIFAR-100 class has 500 original training examples, of which 50 are held out for validation by the repository's stratified split. Therefore **450 allowed TRAIN examples/class** remain. The pilot sets `replay_reservoir_fraction=0.10`, giving **k=45 distinct examples per class per client**. Across the 20 unique classes learned by a client in the disjoint pilot, that is at most **900 private samples**. This replaces, rather than combines with, the previous flat 1,000-example private cap.

The reservoir processes each *previously unprocessed* local training index as it arrives. For class c, with n locally seen distinct examples, append if the stored count is below k. Otherwise draw j uniformly from integers 0 through n-1; replace slot j if j<k. Keep the per-class n and processed-ID ledger through synchronization and disconnected periods. Never let rebroadcast server exemplars consume the private quota or reenter the client's private reservoir as newly observed data. This yields uniform sample inclusion for each class with fixed k, independent of task/arrival batch boundaries.

## Server update and reconnection

1. Every client learns each consecutive binary episode and updates its class-reservoir; all clients learn even when disconnected.
2. Online clients upload accumulated learned model deltas **and their private reservoir samples** as labeled 32×32 uint8 exemplar payloads with training-ID provenance.
3. The server merges the uploaded exemplars into the full deduplicated archive. The archive is *not capped* at 45 samples/class across all clients, because cross-client overlap or changed reservoirs may introduce additional unique samples.
4. The server broadcasts its updated model and **entire archive** to connected clients. These samples are available for replay beginning at the first subsequent local task.
5. Offline clients retain their unsent private model/reservoir and old received archive, and do not upload/download until reconnecting.
6. The last communication-only flush brings all 100 pilot class reservoirs to the server. In the non-overlap pilot, **exactly 4,500 unique exemplars** are expected in the server archive after the flush.

## Run two jobs concurrently

```bash
ssh mtechdeserver
cd Correlation-Aware-FCL
git pull --ff-only origin main
python -m pip install -e '.[test]'
python -m pytest -q
bash scripts/run_primary_probe_cifar100.sh --backbone vit --plan-only
mkdir -p logs
nohup bash scripts/run_primary_probe_cifar100.sh --backbone vit --seed 0 --gpus 0,1 --replay-fraction 0.10 --epochs 10 --lr 0.003 > logs/pilot_reservoir_v9.log 2>&1 < /dev/null &
```

This v9 experiment version cannot reuse prior checkpoint status markers. Results are isolated under `results/primary_cifar100_shared_reservoir_v9_vit_seed0/`. Nothing is claimed to have been remotely launched from this chat. Training samples move to the server; this is an explicit data-sharing experimental variant, **not privacy-preserving federated learning**.
