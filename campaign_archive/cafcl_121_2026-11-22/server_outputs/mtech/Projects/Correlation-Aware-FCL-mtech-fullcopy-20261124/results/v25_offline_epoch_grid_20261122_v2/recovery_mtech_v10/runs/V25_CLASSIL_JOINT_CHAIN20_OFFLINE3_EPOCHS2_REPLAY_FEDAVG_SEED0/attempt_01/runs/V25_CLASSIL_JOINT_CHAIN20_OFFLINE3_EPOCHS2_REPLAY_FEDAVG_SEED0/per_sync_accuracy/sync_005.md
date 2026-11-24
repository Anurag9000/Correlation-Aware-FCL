# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.54% on 26 observed labels.
Validation accuracy: 0.1276923076923077; seen labels never predicted: 5.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.14173913043478262.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **41.0%** (0: 0%, 1: 82%) [PENDING upload] | **15.5%** (5: 1%, 6: 30%) | **0.5%** (10: 1%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] | **6.0%** (20: 12%, 21: 0%) [PENDING upload] |
| Client 1 | **51.5%** (1: 82%, 2: 21%) | **16.5%** (6: 30%, 7: 3%) | **20.0%** (11: 0%, 12: 40%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] | **0.0%** (21: 0%, 22: 0%) [PENDING upload] |
| Client 2 | **13.5%** (2: 21%, 3: 6%) | **3.5%** (7: 3%, 8: 4%) | **20.0%** (12: 40%, 13: 0%) | **12.0%** (17: 0%, 18: 24%) | **23.0%** (22: 0%, 23: 46%) |
| Client 3 | **15.5%** (3: 6%, 4: 25%) | **14.0%** (8: 4%, 9: 24%) | **2.0%** (13: 0%, 14: 4%) | **13.0%** (18: 24%, 19: 2%) | **23.5%** (23: 46%, 24: 1%) |
| Client 4 | **13.0%** (4: 25%, 5: 1%) | **12.5%** (9: 24%, 10: 1%) | **2.0%** (14: 4%, 15: 0%) | **7.0%** (19: 2%, 20: 12%) | **0.5%** (24: 1%, 25: 0%) |
