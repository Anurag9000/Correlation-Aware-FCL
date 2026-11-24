# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.75% on 36 observed labels.
Validation accuracy: 0.13; seen labels never predicted: 20.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.1434375.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **35.0%** (0: 28%, 1: 42%) | **0.0%** (5: 0%, 6: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **0.5%** (20: 1%, 21: 0%) | **0.0%** (25: 0%, 26: 0%) | **72.0%** (30: 91%, 31: 53%) |
| Client 1 | **21.0%** (1: 42%, 2: 0%) | **0.0%** (6: 0%, 7: 0%) | **14.0%** (11: 0%, 12: 28%) | **8.0%** (16: 0%, 17: 16%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] | **26.5%** (31: 53%, 32: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **14.0%** (12: 28%, 13: 0%) | **28.5%** (17: 16%, 18: 41%) | **4.5%** (22: 0%, 23: 9%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **0.0%** (8: 0%, 9: 0%) | **4.0%** (13: 0%, 14: 8%) | **20.5%** (18: 41%, 19: 0%) | **36.0%** (23: 9%, 24: 63%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] | **22.0%** (33: 0%, 34: 44%) [PENDING upload] |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **4.0%** (14: 8%, 15: 0%) | **0.5%** (19: 0%, 20: 1%) | **31.5%** (24: 63%, 25: 0%) | **45.5%** (29: 0%, 30: 91%) | **39.5%** (34: 44%, 35: 35%) |
