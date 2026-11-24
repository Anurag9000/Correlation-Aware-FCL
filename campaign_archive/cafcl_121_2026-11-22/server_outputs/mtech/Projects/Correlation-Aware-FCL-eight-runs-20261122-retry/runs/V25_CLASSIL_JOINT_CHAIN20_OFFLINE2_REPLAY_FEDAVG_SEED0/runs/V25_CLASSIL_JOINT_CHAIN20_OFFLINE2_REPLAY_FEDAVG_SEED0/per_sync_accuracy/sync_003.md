# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 33.38% on 16 observed labels.
Validation accuracy: 0.31875; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.33375.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **46.0%** (0: 55%, 1: 37%) | **47.5%** (5: 64%, 6: 31%) | **8.0%** (10: 16%, 11: 0%) |
| Client 1 | **52.5%** (1: 37%, 2: 68%) | **38.0%** (6: 31%, 7: 45%) | **3.0%** (11: 0%, 12: 6%) |
| Client 2 | **37.0%** (2: 68%, 3: 6%) | **55.5%** (7: 45%, 8: 66%) | **5.5%** (12: 6%, 13: 5%) |
| Client 3 | **23.5%** (3: 6%, 4: 41%) | **54.0%** (8: 66%, 9: 42%) | **28.5%** (13: 5%, 14: 52%) |
| Client 4 | **52.5%** (4: 41%, 5: 64%) | **29.0%** (9: 42%, 10: 16%) | **26.0%** (14: 52%, 15: 0%) |
