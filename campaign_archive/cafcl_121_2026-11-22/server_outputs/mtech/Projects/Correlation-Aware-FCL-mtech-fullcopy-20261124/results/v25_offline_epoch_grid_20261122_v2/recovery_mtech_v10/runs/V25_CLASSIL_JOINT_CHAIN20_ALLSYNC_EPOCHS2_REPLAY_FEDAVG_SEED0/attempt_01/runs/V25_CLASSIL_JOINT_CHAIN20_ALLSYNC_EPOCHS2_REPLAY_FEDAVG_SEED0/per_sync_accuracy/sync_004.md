# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.29% on 21 observed labels.
Validation accuracy: 0.19428571428571428; seen labels never predicted: 3.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.22285714285714286.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **57.5%** (0: 70%, 1: 45%) | **21.5%** (5: 29%, 6: 14%) | **1.0%** (10: 0%, 11: 2%) | **28.5%** (15: 37%, 16: 20%) |
| Client 1 | **40.5%** (1: 45%, 2: 36%) | **9.5%** (6: 14%, 7: 5%) | **4.0%** (11: 2%, 12: 6%) | **52.0%** (16: 20%, 17: 84%) |
| Client 2 | **18.0%** (2: 36%, 3: 0%) | **12.5%** (7: 5%, 8: 20%) | **3.0%** (12: 6%, 13: 0%) | **70.5%** (17: 84%, 18: 57%) |
| Client 3 | **3.5%** (3: 0%, 4: 7%) | **14.0%** (8: 20%, 9: 8%) | **0.0%** (13: 0%, 14: 0%) | **42.5%** (18: 57%, 19: 28%) |
| Client 4 | **18.0%** (4: 7%, 5: 29%) | **4.0%** (9: 8%, 10: 0%) | **18.5%** (14: 0%, 15: 37%) | **14.0%** (19: 28%, 20: 0%) |
