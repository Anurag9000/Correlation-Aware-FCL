# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.70% on 60 observed labels.
Validation accuracy: 0.060333333333333336; seen labels never predicted: 40.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.07125.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) | **0.0%** (50: 0%, 51: 0%) |
| Client 1 | **27.5%** (2: 0%, 3: 55%) | **2.0%** (12: 2%, 13: 2%) | **2.0%** (22: 0%, 23: 4%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] |
| Client 2 | **8.0%** (4: 9%, 5: 7%) | **9.0%** (14: 4%, 15: 14%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **1.5%** (44: 0%, 45: 3%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.5%** (26: 1%, 27: 0%) | **17.0%** (36: 0%, 37: 34%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] |
| Client 4 | **52.5%** (8: 69%, 9: 36%) | **41.0%** (18: 67%, 19: 15%) | **0.0%** (28: 0%, 29: 0%) | **2.0%** (38: 0%, 39: 4%) | **8.0%** (48: 16%, 49: 0%) | **0.0%** (58: 0%, 59: 0%) |
