# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.55% on 11 observed labels.
Validation accuracy: 0.19818181818181818; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.19545454545454546.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **67.5%** (0: 70%, 1: 65%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **33.5%** (1: 65%, 2: 2%) | **6.5%** (6: 0%, 7: 13%) |
| Client 2 | **1.0%** (2: 2%, 3: 0%) | **6.5%** (7: 13%, 8: 0%) |
| Client 3 | **14.0%** (3: 0%, 4: 28%) | **1.0%** (8: 0%, 9: 2%) |
| Client 4 | **14.0%** (4: 28%, 5: 0%) | **18.5%** (9: 2%, 10: 35%) |
