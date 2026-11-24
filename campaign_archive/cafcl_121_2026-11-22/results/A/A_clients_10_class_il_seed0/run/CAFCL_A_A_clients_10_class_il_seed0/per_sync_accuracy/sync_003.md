# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.83% on 60 observed labels.
Validation accuracy: 0.12166666666666667; seen labels never predicted: 19.
Server-accepted classes: 60; pending private classes: 0.
Accepted-only TEST accuracy: 0.12833333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **36.5%** (0: 54%, 1: 19%) | **15.5%** (20: 21%, 21: 10%) | **0.0%** (40: 0%, 41: 0%) |
| Client 1 | **27.5%** (2: 21%, 3: 34%) | **34.5%** (22: 2%, 23: 67%) | **0.0%** (42: 0%, 43: 0%) |
| Client 2 | **24.5%** (4: 26%, 5: 23%) | **9.0%** (24: 17%, 25: 1%) | **0.0%** (44: 0%, 45: 0%) |
| Client 3 | **14.0%** (6: 4%, 7: 24%) | **34.0%** (26: 5%, 27: 63%) | **0.0%** (46: 0%, 47: 0%) |
| Client 4 | **19.5%** (8: 20%, 9: 19%) | **1.0%** (28: 0%, 29: 2%) | **0.0%** (48: 0%, 49: 0%) |
| Client 5 | **1.5%** (10: 0%, 11: 3%) | **31.5%** (30: 46%, 31: 17%) | **0.0%** (50: 0%, 51: 0%) |
| Client 6 | **32.0%** (12: 46%, 13: 18%) | **25.5%** (32: 0%, 33: 51%) | **0.0%** (52: 0%, 53: 0%) |
| Client 7 | **6.5%** (14: 8%, 15: 5%) | **1.5%** (34: 3%, 35: 0%) | **0.0%** (54: 0%, 55: 0%) |
| Client 8 | **47.5%** (16: 17%, 17: 78%) | **0.5%** (36: 1%, 37: 0%) | **0.0%** (56: 0%, 57: 0%) |
| Client 9 | **16.5%** (18: 22%, 19: 11%) | **6.0%** (38: 6%, 39: 6%) | **0.0%** (58: 0%, 59: 0%) |
