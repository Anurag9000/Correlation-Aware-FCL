# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.06% on 16 observed labels.
Validation accuracy: 0.17125; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.170625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **52.5%** (0: 100%, 1: 5%) | **0.0%** (5: 0%, 6: 0%) | **10.5%** (10: 20%, 11: 1%) |
| Client 1 | **8.5%** (1: 5%, 2: 12%) | **0.0%** (6: 0%, 7: 0%) | **0.5%** (11: 1%, 12: 0%) |
| Client 2 | **6.0%** (2: 12%, 3: 0%) | **13.5%** (7: 0%, 8: 27%) | **24.0%** (12: 0%, 13: 48%) |
| Client 3 | **21.5%** (3: 0%, 4: 43%) | **15.0%** (8: 27%, 9: 3%) | **31.0%** (13: 48%, 14: 14%) |
| Client 4 | **21.5%** (4: 43%, 5: 0%) | **11.5%** (9: 3%, 10: 20%) | **7.0%** (14: 14%, 15: 0%) |
