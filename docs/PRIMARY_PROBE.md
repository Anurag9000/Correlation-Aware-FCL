# Canonical CIFAR-100 Replay/FedAvg primary probe

This is the sole current primary FedAvg recipe in the repository. It fixes the model, training views, task stream, optimizer state, server aggregation, and comparison arms to the one-seed run that produced the highest completed matched accuracies in both arms.

## Fixed recipe

| Item | Setting |
|---|---|
| Dataset | Official CIFAR-100, native 32×32; 450 training examples/class |
| Model | CIFAR-stem ResNet-18, random initialization |
| Training views | Four-pixel zero-padded random crop and horizontal flip; train only, including Replay reads |
| Local learner | Replay; 50 examples/class, Algorithm R reservoir |
| Server | Sample-weighted whole-model FedAvg; no classwise head fusion |
| Local optimizer | SGD, lr 0.03, momentum 0.9, weight decay 0.0005 |
| Optimizer state | Momentum persists across local tasks and communication |
| Batch and epochs | Batch 64; 10 epochs/task |
| Clients | 5 independently trained client models |
| Stream | Two 2-class local tasks/client at the first sync, then one 2-class task/client at each of eight further learning syncs |
| Participation | Full 5/5 versus rotating 4/5; offline clients continue local training |
| Closing sync | One communication-only flush; no local training in the flush |
| Seed | 0 |
| Evaluation | All 100 classes after the final flush |

The first four labels/client and later pairs form 100 unique CIFAR-100 classes over nine learning syncs. Each arm runs on one GPU with bounded client execution concurrency 1. The rotating arm's returning client update uses the repository's marked stale-update adapter; therefore it is not synchronous textbook FedAvg. The aggregation equation remains sample-weighted full-model FedAvg, including all classifier rows, without row-specific postprocessing.

## Canonical result

| Participation | Test accuracy | Integrated classes |
|---|---:|---:|
| Full 5/5 | **38.58%** | 100/100 |
| Rotating offline 4/5 | **30.58%** | 100/100 |

Against the immediately preceding matched V19 run without crop/flip, these results improved by 17.92 points for full participation and 15.86 points for rotating offline. Both experiments use seed 0; this is not a multi-seed estimate.

The selected run artifacts are archived at `results/post_v20_cifar_crop_flip_20261120/` on MtechDEServer1. It was executed with the V20 opt-in crop/flip code and persistent local momentum; the V21 primary runner now fixes those exact training semantics as its sole recipe and uses a distinct protocol and output root.

## Run

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The probe intentionally has no switches for optimizer state, augmentation, replay budget, backbone, or seed. Changing any of these creates a different experiment and must not be reported as this canonical result.

Results from a future run of the fixed recipe are written under `results/primary_cifar100_fedavg_v21_resnet18_cropflip_pct11p1111_bs64_seed0/`, including a per-sync report for each arm and a paired class-by-class comparison.

## Interpretation limits

This is a one-seed empirical result. All classes reached the server, but that does not imply all were learned equally well. In the preceding V19 run, classes 90–99 each had only 20% of the full-participation FedAvg example weight attached to a client that updated that row, and all ten had 0% test accuracy. This follows from the fixed disjoint class stream, frozen unseen rows, and canonical whole-model FedAvg. Giving those classifier rows different aggregation weights would change the aggregation rule and is not part of this recipe.

The repository contains other local continual-learning learners and server aggregation research methods. They remain distinct experiments and are not alternate implementations of the primary Replay/FedAvg recipe.
