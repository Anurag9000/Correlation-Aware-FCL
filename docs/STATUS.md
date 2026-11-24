# Project status

## Active Task-IL migration

The one-seed V26 Task-IL CIFAR-100 pair was launched on Mtech from the staged worktree. Its schedule is the adjacent-chain plan in [PRIMARY_PROBE.md](PRIMARY_PROBE.md): 20 learning syncs, one two-class task per client per sync, overlapping neighboring class pairs, 50 private replay examples/class/client, random one-client model-upload omission in the hidden arm, and a terminal communication-only flush. Both arms use five concurrent client workers on GPU 0. Output is under `/dev/shm/cafcl_taskil_v26_seed0/`. The latest successful status snapshot is 08:50 UTC; current running/completion state is unknown because later SSH checks did not pass Tailscale authentication. Accuracy will not be judged before sync 10. V26 predates the source-task replay-routing fix now in the workspace: its replay used all currently known heads containing a label. The updated implementation routes replay to each stored image's originating task head and requires a separate run. Older V25 results below used one global class-indexed head and are not results for this Task-IL design.

The V27 source-task routing implementation is local and has not been staged or run on Mtech. A V26 completion/status snapshot is required before staging it or launching its one-seed pair; no V26 job will be stopped.

The local Task-IL integrated-only diagnostic now filters task heads using source-task IDs for training examples acknowledged by the server, and records the included heads in each sync report. V26 reports created by the earlier reporter may include an unuploaded active head in this diagnostic; recompute or exclude that accepted-only figure before using it for causal interpretation.

A further static audit found and fixed an evaluation crash: the final partial Task-IL batch padded images to 64 but previously supplied task IDs only for unpadded labels. The local evaluator now pads labels alongside images and discards padded predictions. For this schedule, sync 1 evaluates 300 validation examples (six classes × 50), leaving a final 44-example batch; the old call would pass 44 task IDs for 64 images and raise. This implies an early V26 failure if it used that evaluator, but the remote status and traceback remain unverified until SSH access returns.

The latest successful Mtech snapshot remains 08:50 UTC. Status attempts at 09:20, 09:50, 10:20, and 10:50 UTC reached the B550 Tailscale SSH banner but timed out before observing Mtech processes or artifacts; each requires the additional Tailscale SSH check. The 10:50 attempt returned a fresh authentication URL but did not establish an Mtech session. The next status-only attempt is due no earlier than 11:20 UTC. Local V27 plan-only validation succeeded at 10:19 UTC and the separate V28 batch-sync plan validated at 10:58 UTC; neither plan-only command started training.

The current V26/V27 primary-run builder does not enable per-batch synchronization (`stream_sync_every_batch` remains false). I added a separate V28 `--sync-every-batch` runner option and plan audit. It expands the same 20 logical task rounds into 1,630 learning syncs with seed 0; no V28 training was launched. Because per-client task streams have unequal sizes, its plan has 1,580 syncs with five active clients, 20 with two, and 30 with one. Under the fixed one-of-five random outage plan, the omitted client has no current batch on 38 syncs, and the sole active client is omitted on five syncs. Stale pending updates may still be uploaded by other selected clients on those syncs, so this is not a count of empty server aggregations. The treatment's effective active-client participation varies and must be included in any interpretation. Its default partial-batch policy includes same-task padding in the loss to preserve fixed 64-image training batches. An alternate `mask_padding_loss` path masks the loss but still feeds repeated examples to BatchNorm; V28 does not select that alternate path.

Task-IL scoring is class-conditioned on the known schedule: at each sync, test examples use every task head from a client/timestep that has seen their ground-truth class by then. When multiple task IDs contain that class, all are used and their probabilities are averaged by class and renormalized over the union, per the user's clarified scoring rule.

Replay splits each incoming 64-image batch into two updates, each mixing 32 current images with 32 images drawn by class-uniform then image-uniform sampling from the client's available archive. The completed pair report and immutable per-sync reports are in `results/primary_v25_fresh_seed0/` on Mtech.

## Historical results

The 38.58% / 30.58% and V25 46.78% / 42.13% scores predate the Task-IL head migration. They are historical class-IL results only.

## Other suites

The repository also retains distinct local continual-learning methods, factorial research configurations, and server aggregation studies. Dated protocol documents and frozen manifests describe their own historical or separate experiment configurations; they are not instructions for the current primary paired launch.

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
```

## V25 execution note

The fresh run under `results/primary_v25_fresh_seed0` uses its
`run_sequential.sh` wrapper: worker 0 (full participation) finishes before
worker 1 (random-offline) starts. Each arm still trains its five clients with
concurrency 5; the two arms do not overlap in this run. The standard launcher
described in [PRIMARY_PROBE.md](PRIMARY_PROBE.md) starts both arm processes
concurrently. This run-specific change affects GPU scheduling and runtime,
not the per-arm training configuration.


## Completed V25 seed-0 result

The full-participation arm finished at **46.78% TEST / 46.08% validation**; the
random-offline arm finished at **42.13% TEST / 42.12% validation**. Both arms
integrated all 100 classes after 20 learning syncs and the terminal flush. The
paired per-class table is in `results/primary_v25_fresh_seed0/primary_cifar100_fedavg_v25_resnet18_cropflip_classuniform_sampledmixed_replay_chain20_randomoffline_ratio1_bs64_seed0/comparison/`.

At sync 10, full participation scored **42.47% TEST / 43.80% validation**;
random-offline scored **38.78% / 40.59%**. The random-offline arm had 50
server-integrated classes and class 50 pending upload; evaluating only
integrated classes gave 39.56%, so the pending class accounts for only a small
part of that sync's score.

The matched V23 run scored **4.24% full / 5.12% random-offline TEST** at sync
10. The V25 full arm uses the same full-participation archive-delivery behavior,
so that comparison points to the changed replay update path. It does not yet
isolate whether the gain comes from sampling the replay buffer, taking two
optimizer steps per incoming batch, or the associated BatchNorm update pattern.
Those controls are running under the seed-0 ablation root
`results/accuracy_collapse_ablations_v25_seed0_run20261120_2101/`; results are
not available yet.
