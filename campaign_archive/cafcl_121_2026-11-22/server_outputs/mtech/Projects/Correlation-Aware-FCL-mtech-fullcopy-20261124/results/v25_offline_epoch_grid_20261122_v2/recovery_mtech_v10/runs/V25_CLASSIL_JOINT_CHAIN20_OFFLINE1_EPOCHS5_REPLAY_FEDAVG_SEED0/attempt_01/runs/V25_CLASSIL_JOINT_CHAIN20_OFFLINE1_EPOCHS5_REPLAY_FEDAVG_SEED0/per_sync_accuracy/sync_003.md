# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.19% on 16 observed labels.
Validation accuracy: 0.21625; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.211875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **50.0%** (0: 75%, 1: 25%) | **8.5%** (5: 3%, 6: 14%) | **0.5%** (10: 1%, 11: 0%) |
| Client 1 | **21.0%** (1: 25%, 2: 17%) | **24.5%** (6: 14%, 7: 35%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **34.0%** (2: 17%, 3: 51%) | **45.5%** (7: 35%, 8: 56%) | **5.5%** (12: 0%, 13: 11%) |
| Client 3 | **43.0%** (3: 51%, 4: 35%) | **28.0%** (8: 56%, 9: 0%) | **13.5%** (13: 11%, 14: 16%) |
| Client 4 | **19.0%** (4: 35%, 5: 3%) | **0.5%** (9: 0%, 10: 1%) | **8.0%** (14: 16%, 15: 0%) |
