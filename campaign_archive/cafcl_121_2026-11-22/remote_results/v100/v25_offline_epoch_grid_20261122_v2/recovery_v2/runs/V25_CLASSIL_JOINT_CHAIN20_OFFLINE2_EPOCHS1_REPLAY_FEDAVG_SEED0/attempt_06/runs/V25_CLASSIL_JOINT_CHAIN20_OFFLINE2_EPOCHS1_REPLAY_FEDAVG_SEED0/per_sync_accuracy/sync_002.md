# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.64% on 11 observed labels.
Validation accuracy: 0.1709090909090909; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.16636363636363635.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **44.5%** (0: 49%, 1: 40%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **41.0%** (1: 40%, 2: 42%) | **9.5%** (6: 0%, 7: 19%) |
| Client 2 | **21.0%** (2: 42%, 3: 0%) | **9.5%** (7: 19%, 8: 0%) |
| Client 3 | **5.0%** (3: 0%, 4: 10%) | **1.0%** (8: 0%, 9: 2%) |
| Client 4 | **5.0%** (4: 10%, 5: 0%) | **11.5%** (9: 2%, 10: 21%) |
