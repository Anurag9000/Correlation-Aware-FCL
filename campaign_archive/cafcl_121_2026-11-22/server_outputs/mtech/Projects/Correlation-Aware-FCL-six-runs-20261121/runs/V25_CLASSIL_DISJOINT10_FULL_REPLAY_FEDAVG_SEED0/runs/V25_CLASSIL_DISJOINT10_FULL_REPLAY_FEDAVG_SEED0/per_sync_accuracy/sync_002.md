# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.80% on 20 observed labels.
Validation accuracy: 0.112; seen labels never predicted: 7.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.118.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **9.0%** (0: 15%, 1: 3%) | **0.0%** (10: 0%, 11: 0%) |
| Client 1 | **26.0%** (2: 15%, 3: 37%) | **0.0%** (12: 0%, 13: 0%) |
| Client 2 | **25.0%** (4: 44%, 5: 6%) | **0.0%** (14: 0%, 15: 0%) |
| Client 3 | **15.5%** (6: 0%, 7: 31%) | **0.0%** (16: 0%, 17: 0%) |
| Client 4 | **42.5%** (8: 75%, 9: 10%) | **0.0%** (18: 0%, 19: 0%) |
