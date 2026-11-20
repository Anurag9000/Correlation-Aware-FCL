# Four-suite eight-arm pilot campaign

`python scripts/launch_four_fcil_pilots.py --plans-only` validates the four configured pairs without touching GPU state. `--preflight-only` also checks CUDA, GPU 0 availability, free VRAM, and compute-process use. GPU admission and post-smoke free-VRAM thresholds default to zero, so the run can use all available memory. If shared-GPU execution is authorized, `--allow-shared-gpu` permits current occupants and measures their combined peak. The default `--client-training-concurrency 1` trains one of each run's five persistent clients at a time in waves separated by the existing task barriers and reuses one CUDA stream within each run; all eight experiment processes still run concurrently and each physical batch remains 64. Before training, the smoke mirrors this concurrency using each production backbone/loss, five client models, the server model, and native teacher copies; it records peak device and per-process allocator memory. A CUDA out-of-memory failure stops the smoke and identifies the actual capacity limit. The smoke uses random inputs and does not create experiment checkpoints.

Run the supervisor inside a persistent server `tmux` session:

```bash
cd /home/anurag24092/Projects/Correlation-Aware-FCL
.venv/bin/python scripts/launch_four_fcil_pilots.py \
  --seed 0 --lr 0.03 --epochs 10 --gpu 0 \
  --data-root data --wait-for-gpu
```

With `--wait-for-gpu`, the supervisor validates all paired plans first, then checks admission every five minutes against the configured threshold (zero by default). It then runs the eight-process memory smoke and rechecks admission before launching training. It must be started in tmux so the deferred launch survives a client disconnect.

It validates seed-0 plans for GLFC, LGA, FEAT, and FedCLASS reservoir mode, then starts the four suite controllers without waiting between them. Each controller launches its FULL and ROTATING_OFFLINE workers concurrently. The campaign manifest records the commit, plan/config receipts, dataset counts, controller/worker PIDs, and five-minute GPU snapshots. If any controller fails or eight distinct CUDA worker contexts cannot be observed during startup, it terminates the campaign process groups and marks the manifest failed. Re-run the identical commit/config/output path with `--resume` only after a failed campaign; the supervisor verifies prior PIDs have exited and workers resume from per-round checkpoints. Each retry gets a separate log directory and the manifest retains attempt history. A successful campaign requires all four paired controllers to exit successfully; each writes its own comparison artifacts.

The pilot configurations are protocol-harmonized adaptations, not a claim of full paper reproduction. GLFC lacks its full prototype-gradient reconstruction, LGA lacks its full geometric/proxy machinery, and FEAT lacks native leverage selection and geometric correction. See `NATIVE_OFFLINE_FIDELITY.md` and `FedCLASS/pilot/README.md` for method-specific qualifications.

The four suite launch scripts select `CAFCL_PYTHON`, then the repository `.venv/bin/python`, then `python3`. This avoids accidentally invoking a system `python` that may be Python 2.7.
