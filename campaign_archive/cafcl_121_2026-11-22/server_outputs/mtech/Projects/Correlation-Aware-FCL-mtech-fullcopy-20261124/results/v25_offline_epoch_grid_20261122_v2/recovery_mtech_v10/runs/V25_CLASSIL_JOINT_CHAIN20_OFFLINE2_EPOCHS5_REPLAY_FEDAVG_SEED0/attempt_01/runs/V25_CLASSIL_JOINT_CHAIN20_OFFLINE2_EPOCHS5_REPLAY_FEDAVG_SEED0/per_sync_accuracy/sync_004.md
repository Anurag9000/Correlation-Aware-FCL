# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.62% on 21 observed labels.
Validation accuracy: 0.21428571428571427; seen labels never predicted: 2.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.248.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **50.5%** (0: 43%, 1: 58%) | **43.0%** (5: 37%, 6: 49%) | **2.5%** (10: 2%, 11: 3%) | **32.0%** (15: 64%, 16: 0%) |
| Client 1 | **50.0%** (1: 58%, 2: 42%) | **53.0%** (6: 49%, 7: 57%) | **22.0%** (11: 3%, 12: 41%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **27.0%** (2: 42%, 3: 12%) | **34.5%** (7: 57%, 8: 12%) | **22.0%** (12: 41%, 13: 3%) | **0.5%** (17: 0%, 18: 1%) [PENDING upload] |
| Client 3 | **22.5%** (3: 12%, 4: 33%) | **20.0%** (8: 12%, 9: 28%) | **6.0%** (13: 3%, 14: 9%) | **1.0%** (18: 1%, 19: 1%) |
| Client 4 | **35.0%** (4: 33%, 5: 37%) | **15.0%** (9: 28%, 10: 2%) | **36.5%** (14: 9%, 15: 64%) | **1.0%** (19: 1%, 20: 1%) |
