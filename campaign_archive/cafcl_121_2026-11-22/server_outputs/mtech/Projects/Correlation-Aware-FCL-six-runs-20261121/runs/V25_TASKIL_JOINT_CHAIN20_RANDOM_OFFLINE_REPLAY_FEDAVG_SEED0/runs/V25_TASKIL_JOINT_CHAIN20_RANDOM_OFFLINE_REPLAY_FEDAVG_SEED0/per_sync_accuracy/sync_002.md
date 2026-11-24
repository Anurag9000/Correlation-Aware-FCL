# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.64% on 11 observed labels.
Validation accuracy: 0.5363636363636364; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5263636363636364.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **67.0%** (0: 44%, 1: 90%) | **3.5%** (5: 2%, 6: 5%) |
| Client 1 | **49.0%** (1: 90%, 2: 8%) | **52.5%** (6: 5%, 7: 100%) |
| Client 2 | **21.5%** (2: 8%, 3: 35%) | **100.0%** (7: 100%, 8: 100%) |
| Client 3 | **65.0%** (3: 35%, 4: 95%) | **50.0%** (8: 100%, 9: 0%) |
| Client 4 | **48.5%** (4: 95%, 5: 2%) | **50.0%** (9: 0%, 10: 100%) |
