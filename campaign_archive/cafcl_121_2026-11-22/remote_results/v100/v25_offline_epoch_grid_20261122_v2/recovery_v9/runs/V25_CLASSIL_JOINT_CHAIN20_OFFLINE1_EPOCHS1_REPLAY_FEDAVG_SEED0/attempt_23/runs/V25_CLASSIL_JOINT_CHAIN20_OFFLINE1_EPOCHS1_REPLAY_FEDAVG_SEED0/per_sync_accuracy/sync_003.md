# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.06% on 16 observed labels.
Validation accuracy: 0.1625; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.160625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **57.5%** (0: 98%, 1: 17%) | **1.5%** (5: 2%, 6: 1%) | **8.5%** (10: 16%, 11: 1%) |
| Client 1 | **9.5%** (1: 17%, 2: 2%) | **0.5%** (6: 1%, 7: 0%) | **0.5%** (11: 1%, 12: 0%) |
| Client 2 | **3.5%** (2: 2%, 3: 5%) | **15.5%** (7: 0%, 8: 31%) | **11.5%** (12: 0%, 13: 23%) |
| Client 3 | **17.5%** (3: 5%, 4: 30%) | **22.0%** (8: 31%, 9: 13%) | **20.5%** (13: 23%, 14: 18%) |
| Client 4 | **16.0%** (4: 30%, 5: 2%) | **14.5%** (9: 13%, 10: 16%) | **9.0%** (14: 18%, 15: 0%) |
