# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.19% on 21 observed labels.
Validation accuracy: 0.17523809523809525; seen labels never predicted: 4.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.1619047619047619.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **63.0%** (0: 87%, 1: 39%) | **14.0%** (5: 28%, 6: 0%) | **13.5%** (10: 0%, 11: 27%) | **1.0%** (15: 0%, 16: 2%) |
| Client 1 | **22.5%** (1: 39%, 2: 6%) | **0.0%** (6: 0%, 7: 0%) | **14.5%** (11: 27%, 12: 2%) | **34.5%** (16: 2%, 17: 67%) |
| Client 2 | **3.0%** (2: 6%, 3: 0%) | **0.5%** (7: 0%, 8: 1%) | **1.0%** (12: 2%, 13: 0%) | **61.0%** (17: 67%, 18: 55%) |
| Client 3 | **6.5%** (3: 0%, 4: 13%) | **0.5%** (8: 1%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **30.5%** (18: 55%, 19: 6%) |
| Client 4 | **20.5%** (4: 13%, 5: 28%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **6.5%** (19: 6%, 20: 7%) |
