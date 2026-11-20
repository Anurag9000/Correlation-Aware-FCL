# Three native-method CIFAR-100 offline-client pilots

## Entry points

- `bash GLFC/pilot/run.sh --plan-only` or `bash GLFC/pilot/run.sh`
- `bash LGA/pilot/run.sh --plan-only` or `bash LGA/pilot/run.sh`
- `bash FEAT/pilot/run.sh --plan-only` or `bash FEAT/pilot/run.sh`

Install the project first with `python -m pip install -e '.[test]'`.
The default production call runs exactly TWO separate, concurrent OS processes
on physical GPU 0, each with five persistent client models and a server.
The processes launch as a matched pair, and failure of either is fatal.
Each process follows the established v16 first-four-then-two class stream:
nine training intervals, one communication-only final flush, one cycling
offline client per interval in the intervention, and all clients learning
independently even when disconnected. Final inference never receives task IDs.
All training, replay and evaluation forward batches have physical size 64.
The private/client reservoir uses independent per-class Algorithm R with 50
eligible TRAIN images per class; the server shares accepted deduplicated images.

Results are isolated under `results/native_offline_pilots/{glfc,lga,feat}/seedN/`.
Each method writes `design.json`, both workers' logs, per-sync reports,
per-client/model synchronization traces, checkpoints and terminal comparison.

## Original source protection

The original upstream tracked files under `GLFC/src`, `LGA/` (excluding
new `pilot`) and `FEAT/` (excluding new `pilot`) were copied byte-for-byte
and are *not* altered. The native ResNet sources are dynamically imported from
these directories, and native-method loss adaptations live under `src/cafcl`.

## Scientific fidelity / limitations (important)

These are **protocol-harmonized method adaptations, not exact paper
reproductions**. This is explicitly logged in every design manifest.

- **GLFC:** imports the published CBAM ResNet18 feature extractor and adapts
  the published weighted BCE + sigmoid old-teacher loss to sparse, globally
  indexed 100-class independent client streams. The upstream proxy-gradient
  reconstruction, entropy-triggered exemplar herding and encoder-monitor
  pipeline are **not yet integrated**; the common seeded global 50/class
  reservoir is used instead. Do NOT cite this pilot as full GLFC accuracy.
- **LGA:** imports its CBAM ResNet18 and adapts its published weighted BCE +
  old-teacher objective to the same sparse client stream. Its native
  proxy-gradient reconstruction, geometric radius and feature-augmented
  monitoring path are **not yet integrated**. Do NOT cite as full LGA accuracy.
- **FEAT:** imports the published FEAT ResNet18 feature extractor and ports
  the actual GSA feature-prototype distillation objective with the
  source's required FC-weight transpose corrected. The upstream repository
  lacks critical `utils/inc_net.py` and related modules. This pilot uses
  the compatible CNN+linear classifier, and does **not yet** reproduce native
  per-client leverage-based exemplar selection or geometric projection
  correction. Do NOT cite as complete FEAT accuracy.

The three loss/backbone combinations are distinct, and their complete
model-parameter updates use identical federated delayed-delta semantics.
However, they should be reported as **native-loss ablations** until the
missing method-specific components are restored and validated. These pilots
are not proxies for published SOTA baselines.

## Implementation invariants

- Offline means **no upload and no download**, never "skip local SGD."
- Every local episode trains with seen-class competition; future labels
  are never presented as a task-ID oracle at inference.
- Teacher snapshots survive disconnected episodes, and server teacher
  replacement occurs only after accepted synchronization.
- Once offline updates are received, the server fuses accumulated anchor-
  relative deltas with an explicitly noncanonical stale-update adapter,
  not pretend synchronous FedAvg.
- The test set supplies audit metrics only and must not guide optimization.
- A passed CPU plan test does **not** imply GPU/VRAM suitability or
  scientifically acceptable accuracy.

## Four-suite campaign supervision

The eight-arm campaign supervisor and its GPU admission/monitoring policy are documented in `FOUR_SUITE_PILOT_CAMPAIGN.md`. The supervisor runs all three adapted native-loss pilots plus the FedCLASS CE+FedAvg reservoir arm as four concurrent paired controllers.
