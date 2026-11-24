# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.05% on 40 observed labels.
Validation accuracy: 0.0535; seen labels never predicted: 33.
Server-accepted classes: 20; pending private classes: 20.
Accepted-only TEST accuracy: 0.121.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] |
| Client 1 | **10.2%** (4: 41%, 5: 0%, 6: 0%, 7: 0%) | **0.0%** (24: 0%, 25: 0%, 26: 0%, 27: 0%) [PENDING upload] |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) |
| Client 3 | **0.0%** (12: 0%, 13: 0%, 14: 0%, 15: 0%) [PENDING upload] | **0.0%** (32: 0%, 33: 0%, 34: 0%, 35: 0%) [PENDING upload] |
| Client 4 | **13.0%** (16: 0%, 17: 0%, 18: 52%, 19: 0%) | **37.2%** (36: 0%, 37: 0%, 38: 72%, 39: 77%) |
