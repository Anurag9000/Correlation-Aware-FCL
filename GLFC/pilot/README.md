# GLFC — source-fidelity hold

The original GLFC repository is copied byte-for-byte under `GLFC/`, but
the existing pilot is a **partial adapted-loss ablation**, NOT the complete
original published algorithm. Its source-specific mechanisms are inventoried
in [ORIGINAL_SOURCE_FIDELITY_AUDIT.md](../../docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md).

`bash GLFC/pilot/run.sh --plan-only` is safe for configuration inspection.

`bash GLFC/pilot/run.sh` now **refuses** to claim original-method fidelity
or launch expensive experimental training. For a historical incomplete
ablation only, one can pass `--permit-adapted-legacy`, explicitly accepting
that the original algorithm is NOT being run.

The actual paper-faithful port remains to be implemented and independently
validated before launching the requested fully faithful experiments.
