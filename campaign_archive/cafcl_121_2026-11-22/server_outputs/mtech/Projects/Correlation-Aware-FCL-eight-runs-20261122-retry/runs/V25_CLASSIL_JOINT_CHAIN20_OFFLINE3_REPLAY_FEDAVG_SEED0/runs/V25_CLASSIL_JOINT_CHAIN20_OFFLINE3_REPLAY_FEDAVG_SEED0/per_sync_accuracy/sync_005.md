# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.92% on 26 observed labels.
Validation accuracy: 0.17; seen labels never predicted: 11.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.2026086956521739.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **11.5%** (0: 0%, 1: 23%) [PENDING upload] | **31.0%** (5: 62%, 6: 0%) | **10.0%** (10: 20%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] | **6.5%** (20: 13%, 21: 0%) [PENDING upload] |
| Client 1 | **35.0%** (1: 23%, 2: 47%) | **23.5%** (6: 0%, 7: 47%) | **2.5%** (11: 0%, 12: 5%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] | **0.0%** (21: 0%, 22: 0%) [PENDING upload] |
| Client 2 | **44.0%** (2: 47%, 3: 41%) | **38.0%** (7: 47%, 8: 29%) | **3.0%** (12: 5%, 13: 1%) | **12.5%** (17: 0%, 18: 25%) | **0.0%** (22: 0%, 23: 0%) |
| Client 3 | **58.0%** (3: 41%, 4: 75%) | **43.0%** (8: 29%, 9: 57%) | **1.5%** (13: 1%, 14: 2%) | **22.0%** (18: 25%, 19: 19%) | **0.0%** (23: 0%, 24: 0%) |
| Client 4 | **68.5%** (4: 75%, 5: 62%) | **38.5%** (9: 57%, 10: 20%) | **1.0%** (14: 2%, 15: 0%) | **16.0%** (19: 19%, 20: 13%) | **0.0%** (24: 0%, 25: 0%) |
