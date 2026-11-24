# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.43% on 21 observed labels.
Validation accuracy: 0.1295238095238095; seen labels never predicted: 7.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.1305.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **58.5%** (0: 98%, 1: 19%) | **17.5%** (5: 14%, 6: 21%) | **4.0%** (10: 0%, 11: 8%) | **9.0%** (15: 18%, 16: 0%) |
| Client 1 | **9.5%** (1: 19%, 2: 0%) | **10.5%** (6: 21%, 7: 0%) | **12.5%** (11: 8%, 12: 17%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **9.0%** (7: 0%, 8: 18%) | **8.5%** (12: 17%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **19.5%** (3: 0%, 4: 39%) | **9.0%** (8: 18%, 9: 0%) | **4.5%** (13: 0%, 14: 9%) | **0.0%** (18: 0%, 19: 0%) |
| Client 4 | **26.5%** (4: 39%, 5: 14%) | **0.0%** (9: 0%, 10: 0%) | **13.5%** (14: 9%, 15: 18%) | **0.0%** (19: 0%, 20: 0%) |
