# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 39.50% on 24 observed labels.
Validation accuracy: 0.39; seen labels never predicted: 1.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.395.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **61.5%** (0: 82%, 1: 41%) | **59.0%** (6: 55%, 7: 63%) | **46.5%** (12: 60%, 13: 33%) | **1.0%** (18: 2%, 19: 0%) |
| Client 1 | **58.5%** (2: 65%, 3: 52%) | **50.5%** (8: 48%, 9: 53%) | **47.5%** (14: 40%, 15: 55%) | **2.0%** (20: 4%, 21: 0%) |
| Client 2 | **58.0%** (4: 49%, 5: 67%) | **20.0%** (10: 32%, 11: 8%) | **58.0%** (16: 38%, 17: 78%) | **11.5%** (22: 1%, 23: 22%) |
