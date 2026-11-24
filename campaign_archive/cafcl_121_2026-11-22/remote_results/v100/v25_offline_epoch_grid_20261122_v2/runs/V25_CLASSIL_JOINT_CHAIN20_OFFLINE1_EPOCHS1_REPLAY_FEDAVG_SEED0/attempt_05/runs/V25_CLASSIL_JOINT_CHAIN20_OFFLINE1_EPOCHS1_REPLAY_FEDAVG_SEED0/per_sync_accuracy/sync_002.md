# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.09% on 11 observed labels.
Validation accuracy: 0.21454545454545454; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.2309090909090909.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **49.0%** (0: 66%, 1: 32%) | **6.0%** (5: 2%, 6: 10%) |
| Client 1 | **28.0%** (1: 32%, 2: 24%) | **14.5%** (6: 10%, 7: 19%) |
| Client 2 | **27.0%** (2: 24%, 3: 30%) | **9.5%** (7: 19%, 8: 0%) |
| Client 3 | **50.0%** (3: 30%, 4: 70%) | **0.5%** (8: 0%, 9: 1%) |
| Client 4 | **36.0%** (4: 70%, 5: 2%) | **0.5%** (9: 1%, 10: 0%) |
