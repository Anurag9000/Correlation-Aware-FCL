> **SCIENTIFIC FIDELITY HOLD — CURRENT SOURCE PORT.** Original upstream
> files are byte-identical, and selected original methods now execute,
> but the complete GLFC/LGA/FEAT original training/proxy orchestration
> is NOT yet integrated. Default GPU launch remains blocked. Explicit
> `--permit-adapted-legacy` is a declaration of an INCOMPLETE paper
> adaptation, never certification. See
> [the mechanism inventory](ORIGINAL_SOURCE_FIDELITY_AUDIT.md).

# Three paired original-source compatibility pilots

The launchers are `GLFC/pilot/run.sh`, `LGA/pilot/run.sh`, and
`FEAT/pilot/run.sh`. `--plan-only` is safe and available without GPU.
Both FULL and rotating-OFFLINE arms launch in separate CUDA:0 processes.
Five local clients learn in both arms. The only experimental intervention
is a client temporarily unable to upload or download. The nine training
intervals and communication-only tenth flush cover the prescribed
independent CIFAR-100 two-class local streams. Model input batch64 is
mandatory. Source fidelity is recorded separately from class-stream
protocol adaptation. No GPU training has been launched as a newly
certified faithful study.

## Mechanism execution as presently implemented

| Method | Exact executed published source methods | Remaining differences |
|---|---|---|
| GLFC | CBAM feature extractor, source weighted BCE/teacher computation, original FedAvg, entropy`_signal`, `update_new_set`, exemplar reduction & herding, original fl_main crop/flip/jitter transform | Source `train` optimizer/schedule not invoked, source `prototype_mask` and proxy gradient inversion/monitor/model-back not invoked, original network dynamic expansion adapted |
| LGA | Original CBAM, weighted BCE, FedAvg, entropy trigger and herding, fl_main PIL transform | Source `train` and proxy stack missing, `compute_radius`, `f_monitor` and geometry augmentation missing |
| FEAT | Source ResNet backbone, GSA, `FLDataSelector` masked QR/SVD leverage selection, ECC projectors/energy and aggregation, authors' crop/flip/jitter | Official `utils/inc_net.py`, `methods/GSA.py`, `methods/EGC.py` are missing upstream; fixed ETF and correction code are explicitly labelled third-party/paper reconstructions, original `FEAT._fl_train` orchestration not directly executed |

The source selector may choose duplicate FEAT retention tickets, as the
author `torch.multinomial(...replacement=True)` does. GLFC/LGA memory
is **NOT Algorithm R**: the original entropy detector can legitimately
choose *no exemplars* when the rise is below its threshold. It selects
from the previous local pair only on a positive trigger. The server's
original-method archive receives actual selected source exemplars and
source-method state ONLY from clients whose uploads are accepted.
All selected clients receive accepted original-method artifacts; unsent
hidden-client artifacts remain private until reconnection.

## Explicit experimental adaptations

Original authors do not train this particular disjoint, five-client,
two-class, rotating-offline protocol. Dataset indexing and visible-class
head mapping, per-client persistent state, 64-input physical batch
constraint, terminal flush, and off-schedule delayed model fusion
are documented changes. Our delayed update uses a noncanonical
lag-damped, BatchNorm-aware algorithm; it does not claim to be the
authors' original synchronous FedAvg behavior.

The two existing source-v3 GLFC/LGA and source-v4 FEAT job IDs and
output folders are **distinct from earlier V1/V2 results**. No historical
data have been altered. The FEAT fixed simplex ETF and subspace
correction use a clearly identified third-party reconstruction because
the official original implementation was never published.

See [ORIGINAL_SOURCE_FIDELITY_AUDIT.md](ORIGINAL_SOURCE_FIDELITY_AUDIT.md)
for exact source paths, verified coverage, original-file SHA manifest,
remaining blockers and the mandatory release conditions.
