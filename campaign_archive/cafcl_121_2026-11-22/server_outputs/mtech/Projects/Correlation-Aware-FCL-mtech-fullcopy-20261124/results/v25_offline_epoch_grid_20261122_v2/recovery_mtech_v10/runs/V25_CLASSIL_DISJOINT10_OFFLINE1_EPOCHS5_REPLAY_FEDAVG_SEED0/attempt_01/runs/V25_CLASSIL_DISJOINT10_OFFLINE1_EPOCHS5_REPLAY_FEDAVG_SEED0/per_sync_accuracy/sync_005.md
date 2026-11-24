# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.30% on 50 observed labels.
Validation accuracy: 0.1716; seen labels never predicted: 11.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.190625.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **47.5%** (0: 45%, 1: 50%) | **1.5%** (10: 1%, 11: 2%) | **40.5%** (20: 38%, 21: 43%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) |
| Client 1 | **32.0%** (2: 27%, 3: 37%) | **56.0%** (12: 55%, 13: 57%) | **7.0%** (22: 0%, 23: 14%) | **0.0%** (32: 0%, 33: 0%) | **0.0%** (42: 0%, 43: 0%) |
| Client 2 | **31.0%** (4: 34%, 5: 28%) | **14.5%** (14: 18%, 15: 11%) | **28.0%** (24: 52%, 25: 4%) | **1.0%** (34: 2%, 35: 0%) | **0.0%** (44: 0%, 45: 0%) |
| Client 3 | **22.0%** (6: 21%, 7: 23%) | **48.0%** (16: 16%, 17: 80%) | **29.0%** (26: 5%, 27: 53%) | **4.5%** (36: 6%, 37: 3%) | **2.0%** (46: 4%, 47: 0%) |
| Client 4 | **20.5%** (8: 15%, 9: 26%) | **45.0%** (18: 44%, 19: 46%) | **26.0%** (28: 22%, 29: 30%) | **1.5%** (38: 1%, 39: 2%) | **0.0%** (48: 0%, 49: 0%) [PENDING upload] |
