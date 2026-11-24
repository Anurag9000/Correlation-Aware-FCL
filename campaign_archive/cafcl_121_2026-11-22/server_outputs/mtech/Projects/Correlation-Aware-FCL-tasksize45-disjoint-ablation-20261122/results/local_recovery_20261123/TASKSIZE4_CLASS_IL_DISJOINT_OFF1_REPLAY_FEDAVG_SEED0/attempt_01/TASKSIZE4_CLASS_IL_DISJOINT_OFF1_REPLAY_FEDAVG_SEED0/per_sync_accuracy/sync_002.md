# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 4.50% on 40 observed labels.
Validation accuracy: 0.049; seen labels never predicted: 33.
Server-accepted classes: 36; pending private classes: 4.
Accepted-only TEST accuracy: 0.05.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%) | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) |
| Client 1 | **19.5%** (4: 77%, 5: 1%, 6: 0%, 7: 0%) | **0.0%** (24: 0%, 25: 0%, 26: 0%, 27: 0%) |
| Client 2 | **17.0%** (8: 68%, 9: 0%, 10: 0%, 11: 0%) | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] |
| Client 3 | **8.5%** (12: 34%, 13: 0%, 14: 0%, 15: 0%) | **0.0%** (32: 0%, 33: 0%, 34: 0%, 35: 0%) |
| Client 4 | **0.0%** (16: 0%, 17: 0%, 18: 0%, 19: 0%) | **0.0%** (36: 0%, 37: 0%, 38: 0%, 39: 0%) |
