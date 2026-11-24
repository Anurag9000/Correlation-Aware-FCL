# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 31.02% on 50 observed labels.
Validation accuracy: 0.3236; seen labels never predicted: 10.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.3102.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **63.0%** (0: 75%, 1: 51%) | **25.0%** (10: 27%, 11: 23%) | **60.0%** (20: 57%, 21: 63%) | **43.5%** (30: 69%, 31: 18%) | **0.0%** (40: 0%, 41: 0%) |
| Client 1 | **37.5%** (2: 43%, 3: 32%) | **48.5%** (12: 64%, 13: 33%) | **46.0%** (22: 16%, 23: 76%) | **44.0%** (32: 20%, 33: 68%) | **0.0%** (42: 0%, 43: 0%) |
| Client 2 | **31.5%** (4: 32%, 5: 31%) | **31.5%** (14: 34%, 15: 29%) | **45.0%** (24: 74%, 25: 16%) | **13.5%** (34: 25%, 35: 2%) | **0.0%** (44: 0%, 45: 0%) |
| Client 3 | **40.5%** (6: 36%, 7: 45%) | **46.0%** (16: 28%, 17: 64%) | **33.0%** (26: 16%, 27: 50%) | **33.0%** (36: 44%, 37: 22%) | **0.0%** (46: 0%, 47: 0%) |
| Client 4 | **46.5%** (8: 45%, 9: 48%) | **29.5%** (18: 29%, 19: 30%) | **31.5%** (28: 41%, 29: 22%) | **26.5%** (38: 18%, 39: 35%) | **0.0%** (48: 0%, 49: 0%) |
