# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 2.68% on 50 observed labels.
Validation accuracy: 0.0256; seen labels never predicted: 44.
Server-accepted classes: 35; pending private classes: 15.
Accepted-only TEST accuracy: 0.038285714285714284.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **20.2%** (0: 0%, 1: 0%, 2: 1%, 3: 0%, 4: 100%) | **0.0%** (25: 0%, 26: 0%, 27: 0%, 28: 0%, 29: 0%) [PENDING upload] |
| Client 1 | **3.0%** (5: 0%, 6: 0%, 7: 0%, 8: 15%, 9: 0%) | **0.0%** (30: 0%, 31: 0%, 32: 0%, 33: 0%, 34: 0%) |
| Client 2 | **0.0%** (10: 0%, 11: 0%, 12: 0%, 13: 0%, 14: 0%) [PENDING upload] | **0.0%** (35: 0%, 36: 0%, 37: 0%, 38: 0%, 39: 0%) [PENDING upload] |
| Client 3 | **0.0%** (15: 0%, 16: 0%, 17: 0%, 18: 0%, 19: 0%) | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%, 44: 0%) |
| Client 4 | **3.6%** (20: 0%, 21: 0%, 22: 0%, 23: 18%, 24: 0%) | **0.0%** (45: 0%, 46: 0%, 47: 0%, 48: 0%, 49: 0%) |
