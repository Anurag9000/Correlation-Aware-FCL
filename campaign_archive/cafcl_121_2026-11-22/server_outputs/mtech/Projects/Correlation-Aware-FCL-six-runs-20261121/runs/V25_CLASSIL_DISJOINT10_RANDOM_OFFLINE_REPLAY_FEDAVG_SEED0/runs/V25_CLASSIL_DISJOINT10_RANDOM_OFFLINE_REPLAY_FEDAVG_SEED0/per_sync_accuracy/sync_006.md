# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.23% on 60 observed labels.
Validation accuracy: 0.259; seen labels never predicted: 14.
Server-accepted classes: 58; pending private classes: 2.
Accepted-only TEST accuracy: 0.2713793103448276.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **42.0%** (0: 49%, 1: 35%) | **18.0%** (10: 19%, 11: 17%) | **70.5%** (20: 72%, 21: 69%) | **9.0%** (30: 18%, 31: 0%) | **11.5%** (40: 1%, 41: 22%) | **0.0%** (50: 0%, 51: 0%) |
| Client 1 | **36.5%** (2: 27%, 3: 46%) | **62.5%** (12: 66%, 13: 59%) | **49.5%** (22: 27%, 23: 72%) | **47.5%** (32: 16%, 33: 79%) | **10.0%** (42: 20%, 43: 0%) | **0.0%** (52: 0%, 53: 0%) |
| Client 2 | **28.0%** (4: 28%, 5: 28%) | **22.0%** (14: 20%, 15: 24%) | **44.0%** (24: 57%, 25: 31%) | **15.0%** (34: 13%, 35: 17%) | **1.0%** (44: 2%, 45: 0%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **38.5%** (6: 32%, 7: 45%) | **55.0%** (16: 40%, 17: 70%) | **34.5%** (26: 19%, 27: 50%) | **38.5%** (36: 30%, 37: 47%) | **4.0%** (46: 0%, 47: 8%) | **0.0%** (56: 0%, 57: 0%) |
| Client 4 | **58.5%** (8: 60%, 9: 57%) | **23.0%** (18: 17%, 19: 29%) | **38.0%** (28: 45%, 29: 31%) | **30.0%** (38: 24%, 39: 36%) | **0.0%** (48: 0%, 49: 0%) | **0.0%** (58: 0%, 59: 0%) |
