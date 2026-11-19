# Native 32×32 CIFAR-100 backbone registry (v5)

## Exactly two production choices

| CLI | `RunConfig.model` | Architecture | CIFAR input | Default effective batch |
|---|---|---|---|---:|
| `--backbone vit` | `vit_tiny_patch16_32` | Native ViT-Tiny/16, 12 pre-norm Transformer blocks, width 192, 3 attention heads, patch 16, MLP ratio 4 | 3×32×32 | 1024 |
| `--backbone cnn` | `resnet18_cifar32` | Native ResNet-18 BasicBlocks [2,2,2,2], 3×3 stride-1 CIFAR stem, no initial maxpool | 3×32×32 | 1024 |

Both backbones are implemented directly with PyTorch in `src/cafcl/models.py`. All classes map to one 100-row linear classifier. Feature extraction, class-wise server fusion, continual-learning objectives, memory reference gradients, selection/diagnostics, synchronization, and checkpoint recovery use the same model factory. Experimental configs reject CNNs/ViTs outside these two choices; `models.BACKBONES` and `resolve_backbone()` supply a single future model registry entrypoint.

**Important:** The native 32×32 ViT-Tiny/16 has **only four image patch tokens** (2×2) plus one class token. It is still a ViT-Tiny/16, but this is a severe spatial-information bottleneck and differs substantially from 224×224 ViT evaluation; be cautious when interpreting low accuracy or comparing it to ResNet-18. No automatic upsampling or silent patch-size change is allowed.

**Initialization:** Both backbones are **randomly initialized with the controlled experiment seed**, and are fully fine-tuned. Earlier mandatory pretrained DeiT-Tiny/16 at 224×224 has been retired, **not converted** to a native 32×32 pretrained model. This avoids claiming an equivalent ImageNet checkpoint transfer that has not been implemented/validated. Training-from-scratch may require architecture-appropriate optimizer tuning; the old `.03` SGD learning-rate grid is not automatically optimal for ViT.

**Preprocessing:** CIFAR-100 native 32×32 PIL → tensor; CIFAR-100 training statistics mean `(0.5071,0.4867,0.4408)`, std `(0.2675,0.2565,0.2761)` for both backbones. No image resizing or ImageNet normalization. Synthetic fixtures are explicitly separate and never valid production experiment models.

**Batch 1024:** `RunConfig.batch_size=1024` is mandatory for all production CIFAR runs. `micro_batch_size=16` by default controls physical GPU activation memory; example-weighted gradients accumulate to one optimizer update per logical batch. GEM and A-GEM project the logical-batch gradient, not each microbatch independently. Physical microbatches are **not** numerically equivalent for ResNet BatchNorm running statistics, which update per physical batch. This limitation must be reported. A full physical 1024-image GPU batch is not implied.

**ResNet-specific server state:** `BatchNorm.num_batches_tracked` counters differ across clients; the server uses a monotone maximum, not an invalid exact-equality assertion. Floating-point batch statistics still follow the selected aggregation rule. This is a declared federated BatchNorm heuristic and should not be presented as mathematically equivalent to a centralized BatchNorm pass.

**Versioning:** `cifar100_native32_v5` protocol, distinct model-specific job fingerprints/output roots (`results/cifar100_native32_v5_vit` versus `_cnn`), and immutable backbone description in each run. Never resume a DeiT-v4 or SmallCNN-v3 checkpoint into either v5 model. The default full ledger is the ViT; `--backbone cnn` changes the model family, **not** the data or algorithm cross.

### Literature

- Dosovitskiy et al., *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ICLR 2021): ViT design. https://arxiv.org/abs/2010.11929
- He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016): ResNet-18 family. https://arxiv.org/abs/1512.03385

No official CIFAR-100 GPU results for these v5 architectures have been measured here.
