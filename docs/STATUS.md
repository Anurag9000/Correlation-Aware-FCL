# Project status

## Active primary experiment

The current primary CIFAR-100 paired plan is the V23 adjacent-chain protocol in [PRIMARY_PROBE.md](PRIMARY_PROBE.md): 20 learning syncs, one two-class task per client per sync, overlapping neighboring class pairs, 50 private replay examples/class/client, random one-client model-upload omission in the hidden arm, and a terminal communication-only flush. Both arms use five concurrent client workers and run as separate processes on GPU 0. This is the configured design; it is not evidence that a new training run has completed.

The Replay implementation traverses the complete available replay archive for every incoming-batch optimizer update, in batch-64 chunks. The primary report must be produced by two successful V23 worker runs before the protocol has a new empirical result.

## Historical results

The 38.58% full / 30.58% rotating-offline scores were produced by the earlier nine-sync disjoint-class protocol with sampled-minibatch replay. They remain historical and do not validate the current stream or Replay rule.

## Other suites

The repository also retains distinct local continual-learning methods, factorial research configurations, and server aggregation studies. Dated protocol documents and frozen manifests describe their own historical or separate experiment configurations; they are not instructions for the current primary paired launch.

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
```
