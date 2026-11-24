# V100 recovery v8 storage repair

The v7 attempts `attempt_21` and `attempt_16` reached sync 4, then failed when the V100 `/dev/shm` backed Rclone VFS cache had no free metadata space. At the failure snapshot, the cache reported 8.825 GiB total, 42 objects queued, and 4 uploading; the V100 root filesystem also had 0 free bytes. The 21,178 MiB free VRAM remains enough for two V100 workers under the measured 10,000 MiB peak per worker plus a shared 1,024 MiB margin.

The retry uses an isolated Rclone mount and cache under `/run/user/1027`, which had 9.4 GiB free at 04:37 IST. It preserves per-sync metrics, reports, training rounds, data, optimizer, aggregation, seeds, and evaluation. Only the operational checkpoint cadence changes: full-state `checkpoint_latest.pt` writes occur every 20 logical rounds and at the final round, while `model_stage` is emitted only at the final round. The suite config has `checkpoint_every=0`; this interval does not alter any learning update or reported metric. This reduces repeated multi-hundred-MiB checkpoint uploads while retaining a final full-state checkpoint and final model. A failure before round 20 may require a full retry from the documented seed.

The V8 controller reserves 3,500 MiB backing storage per active worker plus a 1,024 MiB reserve and continues to admit by measured VRAM. With 9.4 GiB cache backing free, at most two workers can be admitted; it must wait for uploads to free backing space before admitting more. V7 caches and failed outputs remain untouched and continue flushing in the separate mount.

V8 source patch and scheduler hashes will be added after applying the patch to an isolated copy of both V100 source variants and verifying the manifest transfer.
