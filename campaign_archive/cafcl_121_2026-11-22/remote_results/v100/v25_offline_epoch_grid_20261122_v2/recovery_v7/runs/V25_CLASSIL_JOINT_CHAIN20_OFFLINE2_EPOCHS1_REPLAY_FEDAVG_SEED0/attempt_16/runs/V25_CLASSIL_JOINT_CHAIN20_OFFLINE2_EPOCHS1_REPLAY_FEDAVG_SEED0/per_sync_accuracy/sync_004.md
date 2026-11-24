# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.90% on 21 observed labels.
Validation accuracy: 0.11904761904761904; seen labels never predicted: 6.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.125.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **58.0%** (0: 99%, 1: 17%) | **19.0%** (5: 14%, 6: 24%) | **5.5%** (10: 0%, 11: 11%) | **15.0%** (15: 30%, 16: 0%) |
| Client 1 | **8.5%** (1: 17%, 2: 0%) | **12.0%** (6: 24%, 7: 0%) | **10.0%** (11: 11%, 12: 9%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **1.0%** (7: 0%, 8: 2%) | **4.5%** (12: 9%, 13: 0%) | **0.5%** (17: 0%, 18: 1%) [PENDING upload] |
| Client 3 | **18.5%** (3: 0%, 4: 37%) | **1.0%** (8: 2%, 9: 0%) | **2.5%** (13: 0%, 14: 5%) | **0.5%** (18: 1%, 19: 0%) |
| Client 4 | **25.5%** (4: 37%, 5: 14%) | **0.0%** (9: 0%, 10: 0%) | **17.5%** (14: 5%, 15: 30%) | **0.5%** (19: 0%, 20: 1%) |
