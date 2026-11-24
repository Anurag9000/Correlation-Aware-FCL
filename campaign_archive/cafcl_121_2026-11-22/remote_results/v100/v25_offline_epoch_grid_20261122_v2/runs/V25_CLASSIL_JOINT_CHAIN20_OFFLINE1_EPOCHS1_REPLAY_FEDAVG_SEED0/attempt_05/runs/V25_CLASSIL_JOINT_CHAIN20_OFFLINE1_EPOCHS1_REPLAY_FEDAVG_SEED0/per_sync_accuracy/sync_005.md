# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.77% on 26 observed labels.
Validation accuracy: 0.16307692307692306; seen labels never predicted: 7.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.1744.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **56.5%** (0: 55%, 1: 58%) | **19.0%** (5: 19%, 6: 19%) | **3.0%** (10: 0%, 11: 6%) | **0.0%** (15: 0%, 16: 0%) | **47.5%** (20: 71%, 21: 24%) |
| Client 1 | **34.0%** (1: 58%, 2: 10%) | **9.5%** (6: 19%, 7: 0%) | **36.0%** (11: 6%, 12: 66%) | **1.0%** (16: 0%, 17: 2%) | **16.5%** (21: 24%, 22: 9%) |
| Client 2 | **5.0%** (2: 10%, 3: 0%) | **5.5%** (7: 0%, 8: 11%) | **33.0%** (12: 66%, 13: 0%) | **15.0%** (17: 2%, 18: 28%) | **12.5%** (22: 9%, 23: 16%) |
| Client 3 | **21.0%** (3: 0%, 4: 42%) | **5.5%** (8: 11%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **14.0%** (18: 28%, 19: 0%) | **8.0%** (23: 16%, 24: 0%) |
| Client 4 | **30.5%** (4: 42%, 5: 19%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **35.5%** (19: 0%, 20: 71%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
