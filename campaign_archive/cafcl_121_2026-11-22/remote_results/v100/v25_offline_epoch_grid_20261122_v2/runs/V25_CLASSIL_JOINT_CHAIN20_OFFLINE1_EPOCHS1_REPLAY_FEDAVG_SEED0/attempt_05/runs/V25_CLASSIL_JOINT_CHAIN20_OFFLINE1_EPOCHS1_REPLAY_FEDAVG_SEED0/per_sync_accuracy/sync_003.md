# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.06% on 16 observed labels.
Validation accuracy: 0.19125; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.190625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **52.5%** (0: 99%, 1: 6%) | **0.5%** (5: 0%, 6: 1%) | **1.0%** (10: 2%, 11: 0%) |
| Client 1 | **16.5%** (1: 6%, 2: 27%) | **0.5%** (6: 1%, 7: 0%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **17.0%** (2: 27%, 3: 7%) | **19.5%** (7: 0%, 8: 39%) | **6.0%** (12: 0%, 13: 12%) |
| Client 3 | **26.5%** (3: 7%, 4: 46%) | **34.5%** (8: 39%, 9: 30%) | **24.0%** (13: 12%, 14: 36%) |
| Client 4 | **23.0%** (4: 46%, 5: 0%) | **16.0%** (9: 30%, 10: 2%) | **18.0%** (14: 36%, 15: 0%) |
