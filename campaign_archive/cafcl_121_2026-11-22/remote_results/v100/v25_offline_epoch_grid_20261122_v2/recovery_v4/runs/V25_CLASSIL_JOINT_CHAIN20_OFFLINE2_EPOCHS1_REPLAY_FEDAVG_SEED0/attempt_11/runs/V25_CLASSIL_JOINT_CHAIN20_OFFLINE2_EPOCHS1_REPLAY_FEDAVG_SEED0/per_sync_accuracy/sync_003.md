# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.06% on 16 observed labels.
Validation accuracy: 0.19125; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.180625.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **45.5%** (0: 83%, 1: 8%) | **13.0%** (5: 19%, 6: 7%) | **18.5%** (10: 0%, 11: 37%) |
| Client 1 | **11.5%** (1: 8%, 2: 15%) | **3.5%** (6: 7%, 7: 0%) | **20.0%** (11: 37%, 12: 3%) |
| Client 2 | **7.5%** (2: 15%, 3: 0%) | **1.5%** (7: 0%, 8: 3%) | **14.5%** (12: 3%, 13: 26%) |
| Client 3 | **19.5%** (3: 0%, 4: 39%) | **1.5%** (8: 3%, 9: 0%) | **32.0%** (13: 26%, 14: 38%) |
| Client 4 | **29.0%** (4: 39%, 5: 19%) | **0.0%** (9: 0%, 10: 0%) | **24.5%** (14: 38%, 15: 11%) |
