# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.50% on 40 observed labels.
Validation accuracy: 0.0675; seen labels never predicted: 26.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.065.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **2.0%** (2: 0%, 3: 4%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **38.0%** (4: 75%, 5: 1%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (26: 0%, 27: 0%) |
| Client 4 | **27.0%** (8: 54%, 9: 0%) | **0.0%** (28: 0%, 29: 0%) |
| Client 5 | **0.0%** (10: 0%, 11: 0%) | **0.0%** (30: 0%, 31: 0%) |
| Client 6 | **28.5%** (12: 57%, 13: 0%) | **0.0%** (32: 0%, 33: 0%) |
| Client 7 | **0.0%** (14: 0%, 15: 0%) | **0.0%** (34: 0%, 35: 0%) |
| Client 8 | **14.5%** (16: 0%, 17: 29%) | **0.0%** (36: 0%, 37: 0%) |
| Client 9 | **20.0%** (18: 40%, 19: 0%) | **0.0%** (38: 0%, 39: 0%) |
