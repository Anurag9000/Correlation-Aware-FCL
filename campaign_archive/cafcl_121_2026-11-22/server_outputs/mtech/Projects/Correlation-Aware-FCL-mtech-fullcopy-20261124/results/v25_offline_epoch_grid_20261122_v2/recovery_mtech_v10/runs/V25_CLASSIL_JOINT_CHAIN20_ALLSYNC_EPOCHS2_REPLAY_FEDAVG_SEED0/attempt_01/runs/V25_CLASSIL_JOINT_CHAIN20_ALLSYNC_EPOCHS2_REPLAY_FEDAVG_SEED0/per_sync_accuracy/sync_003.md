# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.56% on 16 observed labels.
Validation accuracy: 0.2375; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.215625.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **30.5%** (0: 45%, 1: 16%) | **25.0%** (5: 34%, 6: 16%) | **17.5%** (10: 4%, 11: 31%) |
| Client 1 | **17.0%** (1: 16%, 2: 18%) | **8.5%** (6: 16%, 7: 1%) | **36.0%** (11: 31%, 12: 41%) |
| Client 2 | **22.0%** (2: 18%, 3: 26%) | **28.0%** (7: 1%, 8: 55%) | **25.0%** (12: 41%, 13: 9%) |
| Client 3 | **30.5%** (3: 26%, 4: 35%) | **27.5%** (8: 55%, 9: 0%) | **11.5%** (13: 9%, 14: 14%) |
| Client 4 | **34.5%** (4: 35%, 5: 34%) | **2.0%** (9: 0%, 10: 4%) | **7.0%** (14: 14%, 15: 0%) |
