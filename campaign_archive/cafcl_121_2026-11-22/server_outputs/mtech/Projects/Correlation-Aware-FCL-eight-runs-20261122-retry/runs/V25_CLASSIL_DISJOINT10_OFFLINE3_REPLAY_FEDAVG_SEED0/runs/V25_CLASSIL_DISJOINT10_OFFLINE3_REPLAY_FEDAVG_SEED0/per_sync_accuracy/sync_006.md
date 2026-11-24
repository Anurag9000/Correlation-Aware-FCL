# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 2.93% on 60 observed labels.
Validation accuracy: 0.031; seen labels never predicted: 50.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.03666666666666667.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) | **0.0%** (50: 0%, 51: 0%) |
| Client 1 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] |
| Client 2 | **28.5%** (4: 57%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **0.0%** (44: 0%, 45: 0%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **0.5%** (36: 0%, 37: 1%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] |
| Client 4 | **49.0%** (8: 89%, 9: 9%) | **10.0%** (18: 20%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) | **0.0%** (38: 0%, 39: 0%) | **0.0%** (48: 0%, 49: 0%) | **0.0%** (58: 0%, 59: 0%) |
