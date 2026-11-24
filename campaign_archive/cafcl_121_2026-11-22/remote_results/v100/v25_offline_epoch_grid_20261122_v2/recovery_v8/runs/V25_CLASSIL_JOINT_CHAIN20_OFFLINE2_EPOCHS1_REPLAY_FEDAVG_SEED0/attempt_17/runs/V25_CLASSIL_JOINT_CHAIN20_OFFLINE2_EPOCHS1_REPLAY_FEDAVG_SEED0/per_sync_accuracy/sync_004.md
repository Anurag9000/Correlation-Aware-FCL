# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 8.00% on 21 observed labels.
Validation accuracy: 0.08095238095238096; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.084.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **50.5%** (0: 100%, 1: 1%) | **2.0%** (5: 1%, 6: 3%) | **3.0%** (10: 0%, 11: 6%) | **1.0%** (15: 2%, 16: 0%) |
| Client 1 | **0.5%** (1: 1%, 2: 0%) | **1.5%** (6: 3%, 7: 0%) | **14.5%** (11: 6%, 12: 23%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **1.0%** (7: 0%, 8: 2%) | **14.5%** (12: 23%, 13: 6%) | **2.0%** (17: 0%, 18: 4%) [PENDING upload] |
| Client 3 | **6.0%** (3: 0%, 4: 12%) | **1.0%** (8: 2%, 9: 0%) | **3.0%** (13: 6%, 14: 0%) | **5.0%** (18: 4%, 19: 6%) |
| Client 4 | **6.5%** (4: 12%, 5: 1%) | **0.0%** (9: 0%, 10: 0%) | **1.0%** (14: 0%, 15: 2%) | **4.0%** (19: 6%, 20: 2%) |
