# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.80% on 20 observed labels.
Validation accuracy: 0.217; seen labels never predicted: 6.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.218.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **61.5%** (0: 76%, 1: 47%) | **61.5%** (0: 76%, 1: 47%) | **0.5%** (10: 0%, 11: 1%) |
| Client 1 | **41.0%** (2: 40%, 3: 42%) | **41.0%** (2: 40%, 3: 42%) | **0.5%** (12: 0%, 13: 1%) |
| Client 2 | **36.0%** (4: 53%, 5: 19%) | **36.0%** (4: 53%, 5: 19%) | **1.0%** (14: 2%, 15: 0%) |
| Client 3 | **32.0%** (6: 28%, 7: 36%) | **32.0%** (6: 28%, 7: 36%) | **0.0%** (16: 0%, 17: 0%) |
| Client 4 | **45.5%** (8: 74%, 9: 17%) | **45.5%** (8: 74%, 9: 17%) | **0.0%** (18: 0%, 19: 0%) |
