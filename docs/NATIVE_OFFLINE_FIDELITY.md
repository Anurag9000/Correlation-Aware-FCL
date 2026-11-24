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
Both FULL and random-OFFLINE arms launch in separate CUDA:0 processes.
Five local clients learn in both arms. At each of 20 learning syncs, the
hidden arm randomly omits one client model upload; all clients continue
training from their local model and last received archive. Only selected
clients receive updated model weights and the current committed replay archive. A communication-only
terminal flush follows the 20 learning syncs. The class stream is the
adjacent two-class chain documented in [PRIMARY_PROBE.md](PRIMARY_PROBE.md).
Each process runs all five clients concurrently (ten client workers across
the two arms), and model input batch64 is mandatory. Source fidelity is
recorded separately from the shared stream protocol. No GPU training has
been launched as a newly certified faithful study.

## Mechanism execution as presently implemented

| Method | Exact executed published source methods | Remaining differences |
|---|---|---|
| GLFC | Original CBAM, source `train`, BCE/teacher loss, FedAvg, entropy detection and herding, source CIFAR augmentation | Proxy gradient generation/reconstruction and `model_back` teacher routing absent; source network growth adapted |
| LGA | Original CBAM, source `train`, weighted BCE, FedAvg, entropy-triggered herding, source augmentation, and `compute_radius` | Gradient inversion, feature proxy `f_monitor`, server teacher/model-back incomplete |
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
two-class, randomly offline protocol. Dataset indexing and visible-class
head mapping, per-client persistent state, 64-input physical batch
constraint, terminal flush, and off-schedule delayed model fusion
are documented changes. Our delayed update uses a noncanonical
lag-damped, BatchNorm-aware algorithm; it does not claim to be the
authors' original synchronous FedAvg behavior.

The source-v4 GLFC/LGA/FEAT job IDs and
output folders are **distinct from earlier V1/V2 results**. No historical
data have been altered. The FEAT fixed simplex ETF and subspace
correction use a clearly identified third-party reconstruction because
the official original implementation was never published.

See [ORIGINAL_SOURCE_FIDELITY_AUDIT.md](ORIGINAL_SOURCE_FIDELITY_AUDIT.md)
for exact source paths, verified coverage, original-file SHA manifest,
remaining blockers and the mandatory release conditions.

**Source-v4 training correction:** The GLFC/LGA local SGD loops are now executed
from the actual unchanged authors' `train` methods (fresh SGD; momentum 0,
weight decay 0.00001; source epoch schedule), rather than persisting the
shared simulated momentum. The design manifest now reports the **executed**
optimizer instead of the inherited generic pilot settings. All three
source-method identities are V4; no previous V2/V3 experiment can be reused.
