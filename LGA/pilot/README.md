# LGA — full vs rotating-offline pilot

Run from the repository root with `bash LGA/pilot/run.sh --plan-only` to inspect the frozen experiment schedule, then `bash LGA/pilot/run.sh` to launch the concurrent two-arm CUDA pilot. Read [method-fidelity caveats](../../docs/NATIVE_OFFLINE_FIDELITY.md) before interpreting accuracy. Both arms train all five clients and differ only in server communication availability.
