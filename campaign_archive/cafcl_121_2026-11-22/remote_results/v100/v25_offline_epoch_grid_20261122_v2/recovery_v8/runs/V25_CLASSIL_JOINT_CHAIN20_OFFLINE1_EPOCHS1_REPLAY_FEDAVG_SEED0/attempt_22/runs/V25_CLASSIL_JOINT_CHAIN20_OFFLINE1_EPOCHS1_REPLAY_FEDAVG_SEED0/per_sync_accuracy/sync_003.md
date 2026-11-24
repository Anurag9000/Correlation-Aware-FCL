# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.69% on 16 observed labels.
Validation accuracy: 0.18375; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.166875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **51.5%** (0: 100%, 1: 3%) | **3.0%** (5: 0%, 6: 6%) | **11.5%** (10: 23%, 11: 0%) |
| Client 1 | **6.5%** (1: 3%, 2: 10%) | **3.5%** (6: 6%, 7: 1%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **13.0%** (2: 10%, 3: 16%) | **12.0%** (7: 1%, 8: 23%) | **1.0%** (12: 0%, 13: 2%) |
| Client 3 | **30.5%** (3: 16%, 4: 45%) | **14.5%** (8: 23%, 9: 6%) | **17.0%** (13: 2%, 14: 32%) |
| Client 4 | **22.5%** (4: 45%, 5: 0%) | **14.5%** (9: 6%, 10: 23%) | **16.0%** (14: 32%, 15: 0%) |
