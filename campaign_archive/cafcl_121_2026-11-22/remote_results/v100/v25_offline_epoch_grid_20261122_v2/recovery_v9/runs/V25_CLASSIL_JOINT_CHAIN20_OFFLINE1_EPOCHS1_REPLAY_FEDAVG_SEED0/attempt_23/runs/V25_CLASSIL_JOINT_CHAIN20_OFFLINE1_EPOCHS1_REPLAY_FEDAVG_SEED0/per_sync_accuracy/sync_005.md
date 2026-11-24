# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.96% on 26 observed labels.
Validation accuracy: 0.1753846153846154; seen labels never predicted: 6.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.166.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **66.5%** (0: 77%, 1: 56%) | **9.5%** (5: 5%, 6: 14%) | **2.0%** (10: 0%, 11: 4%) | **0.0%** (15: 0%, 16: 0%) | **42.0%** (20: 67%, 21: 17%) |
| Client 1 | **42.5%** (1: 56%, 2: 29%) | **7.0%** (6: 14%, 7: 0%) | **43.5%** (11: 4%, 12: 83%) | **0.0%** (16: 0%, 17: 0%) | **9.0%** (21: 17%, 22: 1%) |
| Client 2 | **14.5%** (2: 29%, 3: 0%) | **5.5%** (7: 0%, 8: 11%) | **41.5%** (12: 83%, 13: 0%) | **9.0%** (17: 0%, 18: 18%) | **1.0%** (22: 1%, 23: 1%) |
| Client 3 | **15.5%** (3: 0%, 4: 31%) | **5.5%** (8: 11%, 9: 0%) | **0.5%** (13: 0%, 14: 1%) | **9.0%** (18: 18%, 19: 0%) | **0.5%** (23: 1%, 24: 0%) |
| Client 4 | **18.0%** (4: 31%, 5: 5%) | **0.0%** (9: 0%, 10: 0%) | **0.5%** (14: 1%, 15: 0%) | **33.5%** (19: 0%, 20: 67%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
