# Mandatory pretrained DeiT-Tiny/16 backbone for all CIFAR-100 jobs (v4)

## Standard published architecture

Touvron et al., **Training data-efficient image transformers & distillation through attention**, ICML 2021, DeiT-Tiny (non-distilled single-token classifier), patch size 16, input 224×224, embed dim 192, 12 Transformer blocks, 3 heads, 5.7M model parameters (including the original ImageNet classifier). Reference model: `timm/deit_tiny_patch16_224.fb_in1k`, library release `timm==1.0.31`. Pretrained ImageNet-1K weights are **mandatory**; the final original 1000-class ImageNet head is removed using `num_classes=0`, and a trainable 100-class canonical CIFAR-100 linear head is initialized for the local/global stream.

Sources: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k and https://arxiv.org/abs/2012.12877 . This is the standard DeiT paper backbone, not a home-grown Transformer.

## One model in the entire research registry

`RunConfig.model` and `full_matrix.base()` enforce `deit_tiny_patch16_224.fb_in1k` for all production CIFAR-100 runs, for SGD/GEM/A-GEM/replay/EWC/SI/ER-ACE/DER/DER++/MAS/LwF and all nine aggregation methods. `models.create_model()` fails closed if a CIFAR user requests `smallcnn`, `linear`, or `mlp`; no silent fallback exists. Lightweight `test_linear`, `test_mlp`, and `test_smallcnn` are strictly isolated **synthetic testing stubs**, never supported CIFAR-100 research jobs.

`features` returns the DeiT embedding (192D). `classifier` holds exactly one canonical row per CIFAR-100 class label, no task-specific heads. Active-label loss and frozen classifier rows work with this final head. Feature diagnostics consume the 192-dimensional embeddings; GEM/A-GEM/EWC/SI/MAS/replay/DER/LwF operate on the shared full ViT and the class-indexed head; the nine server algorithms aggregate matching state dictionaries.

## Preprocessing

All CIFAR-100 training/test images: PIL bicubic resize **32×32 → 224×224**, tensor in [0,1], ImageNet mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`. No stochastic augmentation is introduced by this backbone migration. The synthetic test fixture stays 32×32 and is explicitly not a research experiment.

## Determinism, pinned weights, checkpoint safety

`timm==1.0.31` is pinned. The exact pretrained feature-extractor **SHA-256** is saved with each run's `summary.json` and in the results-root `deit_tiny_pretrained_reference.json`; a differing pretrained weight hash or timm version is rejected for all later runs sharing that root. Optionally set `CAFCL_EXPECTED_DEIT_SHA256` to a previously verified digest. No pretrained-weight files are committed to the repository. The simulator protocol version is `cifar100_deit_tiny_v4`, with independent output root `results/cifar100_deit_v4/`, and no resume from old CNN checkpoints. Initial classifier weights remain seed-controlled; pretrained feature weights are fixed across seeds by the model release/checksum.

## Exact effective batch size 512 on limited GPU memory

The configured **logical batch remains 512**. Local training accumulates example-weighted gradients from physical microbatches of **16** with one optimizer step per logical batch, not one per microbatch. GEM/A-GEM compute the current logical-batch gradient before constrained projection and project **once** per logical optimizer step. Historical reference gradients are also accumulated in bounded physical microbatches. A-GEM/GEM memory samples are independently sampled from the local historical pool. Evaluation uses physical batches ≤32; EWC/MAS/DER consolidation loads bounded input chunks.

`RunConfig.micro_batch_size` controls GPU memory, not the scientific update batch size. The legacy local optimizer learning-rate sweep remains registered but was tuned for CNN runs; **ViT-specific tuning and memory/runtime profiling are required** before the 53,805-job sweep. Full-precision fine-tuning of the 224×224 ViT can still exceed some GPUs or be very slow. A suitable production GPU and pretrained checkpoint download/cache are required; the offline development environment cannot verify actual pretrained downloads or full GPU workloads.

## Test and run

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_all_cifar100.sh --plan-only
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh --suite balanced --seed 0 --limit 2
```

Production must not catch pretrained initialization errors to substitute a CNN or random ViT. Use a **new output directory** for changes to the pretrained checkpoint or preprocessing. Some previous algorithm adaptations and official-test leakage issues remain out of scope of this architecture-only migration.
