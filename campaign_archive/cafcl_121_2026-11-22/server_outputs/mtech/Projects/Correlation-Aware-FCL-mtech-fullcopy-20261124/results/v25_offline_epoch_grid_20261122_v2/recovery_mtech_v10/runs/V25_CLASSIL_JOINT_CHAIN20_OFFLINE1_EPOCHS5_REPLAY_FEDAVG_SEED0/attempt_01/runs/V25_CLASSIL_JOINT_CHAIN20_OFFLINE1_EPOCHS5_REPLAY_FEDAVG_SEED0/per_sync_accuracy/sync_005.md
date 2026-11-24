# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.77% on 26 observed labels.
Validation accuracy: 0.2653846153846154; seen labels never predicted: 4.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.2784.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 49%, 1: 60%) | **34.5%** (5: 31%, 6: 38%) | **7.5%** (10: 0%, 11: 15%) | **12.0%** (15: 24%, 16: 0%) | **24.5%** (20: 48%, 21: 1%) |
| Client 1 | **50.0%** (1: 60%, 2: 40%) | **35.0%** (6: 38%, 7: 32%) | **39.5%** (11: 15%, 12: 64%) | **19.5%** (16: 0%, 17: 39%) | **0.5%** (21: 1%, 22: 0%) |
| Client 2 | **45.0%** (2: 40%, 3: 50%) | **35.5%** (7: 32%, 8: 39%) | **50.0%** (12: 64%, 13: 36%) | **46.0%** (17: 39%, 18: 53%) | **7.5%** (22: 0%, 23: 15%) |
| Client 3 | **42.5%** (3: 50%, 4: 35%) | **29.5%** (8: 39%, 9: 20%) | **20.5%** (13: 36%, 14: 5%) | **27.5%** (18: 53%, 19: 2%) | **7.5%** (23: 15%, 24: 0%) |
| Client 4 | **33.0%** (4: 35%, 5: 31%) | **10.0%** (9: 20%, 10: 0%) | **14.5%** (14: 5%, 15: 24%) | **25.0%** (19: 2%, 20: 48%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
