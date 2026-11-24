# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 9.05% on 21 observed labels.
Validation accuracy: 0.09523809523809523; seen labels never predicted: 10.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.10555555555555556.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **45.0%** (0: 0%, 1: 90%) [PENDING upload] | **5.5%** (5: 0%, 6: 11%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] |
| Client 1 | **63.5%** (1: 90%, 2: 37%) | **5.5%** (6: 11%, 7: 0%) | **0.0%** (11: 0%, 12: 0%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **18.5%** (2: 37%, 3: 0%) | **0.5%** (7: 0%, 8: 1%) | **0.0%** (12: 0%, 13: 0%) | **1.0%** (17: 0%, 18: 2%) [PENDING upload] |
| Client 3 | **10.5%** (3: 0%, 4: 21%) | **11.5%** (8: 1%, 9: 22%) | **1.5%** (13: 0%, 14: 3%) | **2.5%** (18: 2%, 19: 3%) |
| Client 4 | **10.5%** (4: 21%, 5: 0%) | **11.0%** (9: 22%, 10: 0%) | **1.5%** (14: 3%, 15: 0%) | **1.5%** (19: 3%, 20: 0%) |
