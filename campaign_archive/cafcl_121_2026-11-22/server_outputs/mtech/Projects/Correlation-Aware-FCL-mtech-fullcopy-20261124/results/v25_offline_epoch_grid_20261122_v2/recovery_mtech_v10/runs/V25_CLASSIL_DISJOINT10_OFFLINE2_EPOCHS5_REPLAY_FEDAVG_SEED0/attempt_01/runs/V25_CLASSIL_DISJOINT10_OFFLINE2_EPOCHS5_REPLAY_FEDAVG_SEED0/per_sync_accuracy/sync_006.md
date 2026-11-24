# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.38% on 60 observed labels.
Validation accuracy: 0.21166666666666667; seen labels never predicted: 18.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.27795454545454545.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **65.5%** (0: 74%, 1: 57%) | **6.5%** (10: 8%, 11: 5%) | **37.0%** (20: 32%, 21: 42%) | **49.5%** (30: 60%, 31: 39%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] | **0.0%** (50: 0%, 51: 0%) [PENDING upload] |
| Client 1 | **37.5%** (2: 38%, 3: 37%) | **33.5%** (12: 27%, 13: 40%) | **46.0%** (22: 11%, 23: 81%) | **12.0%** (32: 0%, 33: 24%) | **10.0%** (42: 5%, 43: 15%) | **8.0%** (52: 0%, 53: 16%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **34.0%** (6: 30%, 7: 38%) | **30.0%** (16: 15%, 17: 45%) | **33.0%** (26: 7%, 27: 59%) | **31.5%** (36: 16%, 37: 47%) | **35.5%** (46: 5%, 47: 66%) | **0.0%** (56: 0%, 57: 0%) |
| Client 4 | **37.5%** (8: 28%, 9: 47%) | **21.5%** (18: 24%, 19: 19%) | **33.0%** (28: 33%, 29: 33%) | **22.5%** (38: 38%, 39: 7%) | **27.0%** (48: 15%, 49: 39%) | **0.5%** (58: 1%, 59: 0%) |
