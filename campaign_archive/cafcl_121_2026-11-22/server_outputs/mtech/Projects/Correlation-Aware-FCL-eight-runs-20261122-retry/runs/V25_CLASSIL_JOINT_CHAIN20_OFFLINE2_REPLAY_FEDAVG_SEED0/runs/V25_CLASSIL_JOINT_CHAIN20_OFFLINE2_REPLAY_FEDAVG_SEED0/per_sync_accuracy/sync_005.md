# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.88% on 26 observed labels.
Validation accuracy: 0.2723076923076923; seen labels never predicted: 5.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.28884615384615386.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **56.0%** (0: 60%, 1: 52%) | **49.0%** (5: 65%, 6: 33%) | **17.0%** (10: 6%, 11: 28%) | **29.5%** (15: 59%, 16: 0%) | **2.0%** (20: 4%, 21: 0%) |
| Client 1 | **46.5%** (1: 52%, 2: 41%) | **33.5%** (6: 33%, 7: 34%) | **45.0%** (11: 28%, 12: 62%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **49.0%** (2: 41%, 3: 57%) | **39.0%** (7: 34%, 8: 44%) | **39.5%** (12: 62%, 13: 17%) | **0.5%** (17: 0%, 18: 1%) | **0.0%** (22: 0%, 23: 0%) |
| Client 3 | **52.0%** (3: 57%, 4: 47%) | **48.5%** (8: 44%, 9: 53%) | **28.5%** (13: 17%, 14: 40%) | **1.5%** (18: 1%, 19: 2%) | **23.0%** (23: 0%, 24: 46%) |
| Client 4 | **56.0%** (4: 47%, 5: 65%) | **29.5%** (9: 53%, 10: 6%) | **49.5%** (14: 40%, 15: 59%) | **3.0%** (19: 2%, 20: 4%) | **23.0%** (24: 46%, 25: 0%) |
