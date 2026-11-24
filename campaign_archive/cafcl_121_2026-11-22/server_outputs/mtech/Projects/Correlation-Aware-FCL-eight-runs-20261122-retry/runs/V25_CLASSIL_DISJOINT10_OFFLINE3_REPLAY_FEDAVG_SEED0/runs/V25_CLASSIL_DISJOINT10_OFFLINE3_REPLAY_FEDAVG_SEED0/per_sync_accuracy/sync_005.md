# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 4.32% on 50 observed labels.
Validation accuracy: 0.0468; seen labels never predicted: 37.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.06352941176470588.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] | **0.0%** (30: 0%, 31: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] |
| Client 2 | **31.0%** (4: 62%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **0.0%** (44: 0%, 45: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **0.0%** (36: 0%, 37: 0%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] |
| Client 4 | **54.0%** (8: 92%, 9: 16%) | **23.0%** (18: 40%, 19: 6%) | **0.0%** (28: 0%, 29: 0%) | **0.0%** (38: 0%, 39: 0%) | **0.0%** (48: 0%, 49: 0%) |
