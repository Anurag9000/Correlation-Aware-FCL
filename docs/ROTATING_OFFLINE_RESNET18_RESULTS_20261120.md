# CIFAR-100 ResNet-18 rotating-offline results (2026-11-20)

Raw run records, per-sync accuracy reports, configs, and summaries are archived under `results/rotating_offline_resnet18_cifar100_20261120/`. Checkpoint/model files (`.pt`, `.pth`) are intentionally excluded. `SHA256SUMS` records the included payload.

All runs used native 32x32 CIFAR-100 ResNet-18, five clients, seed 0, replay reservoir fraction 0.11111111 (50 images per class), physical batch 64, and ten sync intervals. The v17 and v18 runs used mixed replay and persistent local SGD momentum from v16. v17 added lag damping and stale BatchNorm-buffer protection; v18 additionally capped a single stale client's terminal-flush contribution to one of five client shares.

| Run | Final test accuracy | Final validation accuracy | Notes |
|---|---:|---:|---|
| v15 full baseline | 16.10% | 16.10% | Earlier baseline; no rotating offline client |
| v15 rotating offline | 1.00% | 1.00% | Earlier stale-update behavior |
| v17 rotating offline | 9.23% | 9.66% | Terminal stale upload dropped test accuracy 5.22 points |
| v18 rotating offline | 12.52% | 13.10% | Terminal-flush safeguard reduced the final drop to 0.79 points |

The v18 final accuracy remains below the v15 full baseline. Its per-sync test/validation trajectory is recorded in the archived `metrics.json` and `per_sync_accuracy/` reports. The 90–99 class group remained at 0% in both the old baseline and v18, so the terminal-flush fix does not resolve the last-stage learning weakness.

The v18 guard is implemented in `src/cafcl/simulator.py`; metrics identify upload staleness, damping scale, aggregation family, and terminal guard scale. These results are empirical for this single seed and configuration and do not establish general performance.
