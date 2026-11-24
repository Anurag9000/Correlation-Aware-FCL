# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.35% on 26 observed labels.
Validation accuracy: 0.17461538461538462; seen labels never predicted: 6.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.1804.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **63.0%** (0: 67%, 1: 59%) | **19.0%** (5: 14%, 6: 24%) | **0.5%** (10: 0%, 11: 1%) | **0.0%** (15: 0%, 16: 0%) | **43.5%** (20: 60%, 21: 27%) |
| Client 1 | **40.5%** (1: 59%, 2: 22%) | **12.0%** (6: 24%, 7: 0%) | **41.0%** (11: 1%, 12: 81%) | **3.0%** (16: 0%, 17: 6%) | **17.0%** (21: 27%, 22: 7%) |
| Client 2 | **11.0%** (2: 22%, 3: 0%) | **6.0%** (7: 0%, 8: 12%) | **40.5%** (12: 81%, 13: 0%) | **12.5%** (17: 6%, 18: 19%) | **8.0%** (22: 7%, 23: 9%) |
| Client 3 | **18.0%** (3: 0%, 4: 36%) | **6.0%** (8: 12%, 9: 0%) | **3.5%** (13: 0%, 14: 7%) | **9.5%** (18: 19%, 19: 0%) | **4.5%** (23: 9%, 24: 0%) |
| Client 4 | **25.0%** (4: 36%, 5: 14%) | **0.0%** (9: 0%, 10: 0%) | **3.5%** (14: 7%, 15: 0%) | **30.0%** (19: 0%, 20: 60%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
