# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.84% on 50 observed labels.
Validation accuracy: 0.0668; seen labels never predicted: 30.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.10058823529411764.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] | **0.0%** (30: 0%, 31: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **27.0%** (2: 0%, 3: 54%) | **0.0%** (12: 0%, 13: 0%) | **1.5%** (22: 0%, 23: 3%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] |
| Client 2 | **11.0%** (4: 16%, 5: 6%) | **11.0%** (14: 7%, 15: 15%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **1.0%** (44: 0%, 45: 2%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **28.0%** (36: 0%, 37: 56%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] |
| Client 4 | **43.5%** (8: 51%, 9: 36%) | **47.0%** (18: 74%, 19: 20%) | **0.0%** (28: 0%, 29: 0%) | **0.5%** (38: 0%, 39: 1%) | **0.5%** (48: 1%, 49: 0%) |
