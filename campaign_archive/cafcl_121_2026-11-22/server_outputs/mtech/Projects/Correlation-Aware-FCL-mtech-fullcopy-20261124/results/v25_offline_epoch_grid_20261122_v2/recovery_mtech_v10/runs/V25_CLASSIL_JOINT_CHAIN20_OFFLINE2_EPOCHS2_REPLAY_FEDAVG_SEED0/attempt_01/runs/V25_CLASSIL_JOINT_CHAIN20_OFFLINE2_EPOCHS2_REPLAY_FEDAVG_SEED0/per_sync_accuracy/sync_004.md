# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.86% on 21 observed labels.
Validation accuracy: 0.23238095238095238; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.2295.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **61.5%** (0: 63%, 1: 60%) | **39.0%** (5: 44%, 6: 34%) | **3.5%** (10: 0%, 11: 7%) | **13.0%** (15: 26%, 16: 0%) |
| Client 1 | **35.0%** (1: 60%, 2: 10%) | **18.0%** (6: 34%, 7: 2%) | **36.5%** (11: 7%, 12: 66%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **8.0%** (2: 10%, 3: 6%) | **6.0%** (7: 2%, 8: 10%) | **35.0%** (12: 66%, 13: 4%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **17.0%** (3: 6%, 4: 28%) | **5.0%** (8: 10%, 9: 0%) | **12.0%** (13: 4%, 14: 20%) | **29.5%** (18: 0%, 19: 59%) |
| Client 4 | **36.0%** (4: 28%, 5: 44%) | **0.0%** (9: 0%, 10: 0%) | **23.0%** (14: 20%, 15: 26%) | **39.5%** (19: 59%, 20: 20%) |
