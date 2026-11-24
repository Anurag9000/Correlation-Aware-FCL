# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.91% on 11 observed labels.
Validation accuracy: 0.18; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.1690909090909091.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.0%** (0: 34%, 1: 68%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **35.0%** (1: 68%, 2: 2%) | **0.0%** (6: 0%, 7: 0%) |
| Client 2 | **1.0%** (2: 2%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **16.5%** (3: 0%, 4: 33%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **16.5%** (4: 33%, 5: 0%) | **24.5%** (9: 0%, 10: 49%) |
