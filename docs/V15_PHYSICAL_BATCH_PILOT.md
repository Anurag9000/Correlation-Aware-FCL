> **Historical primary settings:** The current primary Replay/FedAvg recipe is [PRIMARY_PROBE.md](PRIMARY_PROBE.md). This v15 pilot note is retained as history and its launch examples are retired.

# v15 two-job pilot: actual physical batch 64

The pilot was previously configured with effective batch 64 and physical microbatch 8, which had not been requested and changes ResNet-18 BatchNorm behavior. This correction removes that intervention.

- Both concurrent arms, baseline and rotating-offline, use `batch_size=64`, `micro_batch_size=64`, **one forward/backward pass per 64-example SGD update**, and no accumulation across batches. The final partial batch, if any, uses its actual size.
- This is enforced in `make_jobs`: `micro_batch_size=batch_size`. The separate `--micro-batch-size` option is removed from the pilot launcher.
- Models, method, class schedule, FedAvg, replay fraction, SGD momentum/weight decay, epochs, all-server-sync accuracy tables and nine training syncs plus one final flush remain unchanged.
- These settings are a scientifically material change because BatchNorm statistics now use the actual full physical batch. Pilot job IDs and results roots use v15 and do not reuse previous results.
- Scope: the **two-run pilot only**. The 32,133-job factorial registries and historical v14 manifests remain unchanged; altering the entire factorial matrix would require new manifests and an independently audited version.

Run the two concurrent ResNet-18 jobs on the server:

```bash
git pull --ff-only origin main
python -m pip install -e '.[test]'
pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpu 0 --plan-only

mkdir -p logs
nohup bash scripts/run_primary_probe_cifar100.sh \
  --backbone cnn --gpu 0 --seed 0 \
  --batch-size 64 --epochs 10 --lr 0.03 \
  > logs/resnet18_replay_physical64_v15.log 2>&1 < /dev/null &
```

No remote GPU launch is claimed.
