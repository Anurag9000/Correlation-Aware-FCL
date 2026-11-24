# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.76% on 50 observed labels.
Validation accuracy: 0.1664; seen labels never predicted: 21.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.22052631578947368.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **47.5%** (0: 47%, 1: 48%) | **0.0%** (10: 0%, 11: 0%) | **1.0%** (20: 2%, 21: 0%) | **11.5%** (30: 23%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **41.0%** (2: 21%, 3: 61%) | **46.5%** (12: 61%, 13: 32%) | **40.5%** (22: 5%, 23: 76%) | **0.0%** (32: 0%, 33: 0%) | **1.0%** (42: 2%, 43: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] |
| Client 3 | **32.5%** (6: 34%, 7: 31%) | **56.5%** (16: 35%, 17: 78%) | **26.0%** (26: 9%, 27: 43%) | **1.5%** (36: 1%, 37: 2%) | **5.0%** (46: 10%, 47: 0%) |
| Client 4 | **35.0%** (8: 47%, 9: 23%) | **52.0%** (18: 61%, 19: 43%) | **21.0%** (28: 21%, 29: 21%) | **0.5%** (38: 1%, 39: 0%) | **0.0%** (48: 0%, 49: 0%) |
