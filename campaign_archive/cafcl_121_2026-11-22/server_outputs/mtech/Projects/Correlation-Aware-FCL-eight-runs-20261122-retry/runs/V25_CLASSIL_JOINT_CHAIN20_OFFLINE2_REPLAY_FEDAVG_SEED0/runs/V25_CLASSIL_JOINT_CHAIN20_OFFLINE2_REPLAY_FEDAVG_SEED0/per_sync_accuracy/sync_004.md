# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.62% on 21 observed labels.
Validation accuracy: 0.26285714285714284; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.3005.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **58.0%** (0: 54%, 1: 62%) | **55.5%** (5: 59%, 6: 52%) | **11.5%** (10: 13%, 11: 10%) | **10.0%** (15: 20%, 16: 0%) |
| Client 1 | **61.0%** (1: 62%, 2: 60%) | **45.5%** (6: 52%, 7: 39%) | **14.5%** (11: 10%, 12: 19%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **34.5%** (2: 60%, 3: 9%) | **41.5%** (7: 39%, 8: 44%) | **9.5%** (12: 19%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **39.5%** (3: 9%, 4: 70%) | **45.5%** (8: 44%, 9: 47%) | **21.5%** (13: 0%, 14: 43%) | **0.0%** (18: 0%, 19: 0%) |
| Client 4 | **64.5%** (4: 70%, 5: 59%) | **30.0%** (9: 47%, 10: 13%) | **31.5%** (14: 43%, 15: 20%) | **0.0%** (19: 0%, 20: 0%) |
