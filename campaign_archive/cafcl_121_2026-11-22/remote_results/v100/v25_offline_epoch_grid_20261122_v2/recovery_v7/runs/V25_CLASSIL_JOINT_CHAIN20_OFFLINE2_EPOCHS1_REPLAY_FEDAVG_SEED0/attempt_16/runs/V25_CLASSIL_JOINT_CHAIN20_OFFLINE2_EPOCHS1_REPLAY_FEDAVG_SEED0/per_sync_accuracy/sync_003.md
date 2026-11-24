# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.50% on 16 observed labels.
Validation accuracy: 0.23; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.215.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **54.0%** (0: 89%, 1: 19%) | **30.0%** (5: 37%, 6: 23%) | **6.5%** (10: 0%, 11: 13%) |
| Client 1 | **12.5%** (1: 19%, 2: 6%) | **11.5%** (6: 23%, 7: 0%) | **6.5%** (11: 13%, 12: 0%) |
| Client 2 | **3.0%** (2: 6%, 3: 0%) | **3.5%** (7: 0%, 8: 7%) | **13.5%** (12: 0%, 13: 27%) |
| Client 3 | **27.0%** (3: 0%, 4: 54%) | **4.5%** (8: 7%, 9: 2%) | **35.0%** (13: 27%, 14: 43%) |
| Client 4 | **45.5%** (4: 54%, 5: 37%) | **1.0%** (9: 2%, 10: 0%) | **33.5%** (14: 43%, 15: 24%) |
