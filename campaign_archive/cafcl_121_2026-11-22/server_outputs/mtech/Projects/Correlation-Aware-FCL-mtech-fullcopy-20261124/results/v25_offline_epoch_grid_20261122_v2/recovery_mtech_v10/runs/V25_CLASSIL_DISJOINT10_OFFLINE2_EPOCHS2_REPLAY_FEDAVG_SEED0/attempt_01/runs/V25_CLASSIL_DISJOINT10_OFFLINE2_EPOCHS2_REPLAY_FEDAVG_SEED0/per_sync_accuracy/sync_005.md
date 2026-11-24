# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.36% on 50 observed labels.
Validation accuracy: 0.1132; seen labels never predicted: 22.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.14947368421052631.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **56.5%** (0: 75%, 1: 38%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **40.5%** (2: 20%, 3: 61%) | **12.0%** (12: 24%, 13: 0%) | **37.0%** (22: 3%, 23: 71%) | **0.0%** (32: 0%, 33: 0%) | **3.5%** (42: 7%, 43: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] |
| Client 3 | **13.5%** (6: 25%, 7: 2%) | **36.0%** (16: 7%, 17: 65%) | **17.5%** (26: 0%, 27: 35%) | **14.5%** (36: 24%, 37: 5%) | **9.5%** (46: 19%, 47: 0%) |
| Client 4 | **13.5%** (8: 13%, 9: 14%) | **22.0%** (18: 44%, 19: 0%) | **8.0%** (28: 0%, 29: 16%) | **0.0%** (38: 0%, 39: 0%) | **0.0%** (48: 0%, 49: 0%) |
