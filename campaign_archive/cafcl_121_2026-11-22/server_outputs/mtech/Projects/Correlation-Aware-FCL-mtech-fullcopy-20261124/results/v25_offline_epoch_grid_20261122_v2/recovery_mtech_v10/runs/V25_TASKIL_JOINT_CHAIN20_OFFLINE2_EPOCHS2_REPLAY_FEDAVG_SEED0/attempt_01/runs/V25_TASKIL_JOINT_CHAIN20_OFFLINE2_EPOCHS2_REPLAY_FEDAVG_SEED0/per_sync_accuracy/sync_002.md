# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.18% on 11 observed labels.
Validation accuracy: 0.5163636363636364; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5863636363636363.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **47.0%** (0: 86%, 1: 8%) | **58.0%** (5: 21%, 6: 95%) |
| Client 1 | **53.5%** (1: 8%, 2: 99%) | **48.0%** (6: 95%, 7: 1%) |
| Client 2 | **93.0%** (2: 99%, 3: 87%) | **7.5%** (7: 1%, 8: 14%) |
| Client 3 | **75.5%** (3: 87%, 4: 64%) | **56.5%** (8: 14%, 9: 99%) |
| Client 4 | **42.5%** (4: 64%, 5: 21%) | **49.5%** (9: 99%, 10: 0%) |
