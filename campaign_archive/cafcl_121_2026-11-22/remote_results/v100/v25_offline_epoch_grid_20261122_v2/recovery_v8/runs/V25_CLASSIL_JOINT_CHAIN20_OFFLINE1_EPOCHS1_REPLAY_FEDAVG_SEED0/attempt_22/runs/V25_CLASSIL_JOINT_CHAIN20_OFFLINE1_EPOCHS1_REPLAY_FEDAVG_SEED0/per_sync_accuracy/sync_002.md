# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.00% on 11 observed labels.
Validation accuracy: 0.2109090909090909; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.24.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **42.0%** (0: 75%, 1: 9%) | **2.5%** (5: 1%, 6: 4%) |
| Client 1 | **33.0%** (1: 9%, 2: 57%) | **2.0%** (6: 4%, 7: 0%) |
| Client 2 | **33.5%** (2: 57%, 3: 10%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **42.5%** (3: 10%, 4: 75%) | **16.5%** (8: 0%, 9: 33%) |
| Client 4 | **38.0%** (4: 75%, 5: 1%) | **16.5%** (9: 33%, 10: 0%) |
