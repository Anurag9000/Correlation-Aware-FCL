# Four-suite eight-arm pilot campaign

`python scripts/launch_four_fcil_pilots.py --plans-only` validates the four configured pairs without touching GPU state. `--preflight-only` also checks CUDA, GPU 0 availability, free VRAM, and that no other compute process is using that GPU. The production run refuses to proceed if any unrelated GPU process exists or fewer than 28,672 MiB are free. This is a conservative admission gate, not a substitute for measured peak memory from a completed campaign.

Run the supervisor inside a persistent server `tmux` session:

```bash
cd /home/anurag24092/Projects/Correlation-Aware-FCL
.venv/bin/python scripts/launch_four_fcil_pilots.py \
  --seed 0 --lr 0.03 --epochs 10 --gpu 0 \
  --data-root data --wait-for-gpu
```

With `--wait-for-gpu`, the supervisor validates all paired plans first, then checks admission every five minutes until GPU 0 has no unrelated compute process and at least 28 GiB free. It must be started in tmux so the deferred launch survives a client disconnect.

It validates seed-0 plans for GLFC, LGA, FEAT, and FedCLASS reservoir mode, then starts the four suite controllers without waiting between them. Each controller launches its FULL and ROTATING_OFFLINE workers concurrently. The campaign manifest records the commit, plan/config receipts, dataset counts, controller/worker PIDs, and five-minute GPU snapshots. If any controller fails or eight distinct CUDA worker contexts cannot be observed during startup, it terminates the campaign process groups and marks the manifest failed. Re-run the identical commit/config/output path with `--resume` only after a failed campaign; the supervisor verifies prior PIDs have exited and workers resume from per-round checkpoints. Each retry gets a separate log directory and the manifest retains attempt history. A successful campaign requires all four paired controllers to exit successfully; each writes its own comparison artifacts.

The pilot configurations are protocol-harmonized adaptations, not a claim of full paper reproduction. GLFC lacks its full prototype-gradient reconstruction, LGA lacks its full geometric/proxy machinery, and FEAT lacks native leverage selection and geometric correction. See `NATIVE_OFFLINE_FIDELITY.md` and `FedCLASS/pilot/README.md` for method-specific qualifications.

The four suite launch scripts select `CAFCL_PYTHON`, then the repository `.venv/bin/python`, then `python3`. This avoids accidentally invoking a system `python` that may be Python 2.7.
