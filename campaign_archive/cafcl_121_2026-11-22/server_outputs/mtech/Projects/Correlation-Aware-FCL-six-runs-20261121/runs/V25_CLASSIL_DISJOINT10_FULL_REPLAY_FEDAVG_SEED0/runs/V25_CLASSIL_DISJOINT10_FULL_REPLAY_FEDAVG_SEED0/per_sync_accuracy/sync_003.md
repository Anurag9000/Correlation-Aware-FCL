# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.20% on 30 observed labels.
Validation accuracy: 0.21666666666666667; seen labels never predicted: 10.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.232.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **53.5%** (0: 55%, 1: 52%) | **6.5%** (10: 13%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **34.5%** (2: 28%, 3: 41%) | **40.5%** (12: 54%, 13: 27%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **64.0%** (4: 57%, 5: 71%) | **17.0%** (14: 17%, 15: 17%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **36.5%** (6: 32%, 7: 41%) | **42.5%** (16: 21%, 17: 64%) | **0.5%** (26: 1%, 27: 0%) |
| Client 4 | **39.0%** (8: 50%, 9: 28%) | **13.5%** (18: 23%, 19: 4%) | **0.0%** (28: 0%, 29: 0%) |
