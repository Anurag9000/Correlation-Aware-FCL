# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.83% on 60 observed labels.
Validation accuracy: 0.06; seen labels never predicted: 39.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.07291666666666667.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) | **0.0%** (50: 0%, 51: 0%) |
| Client 1 | **26.0%** (2: 1%, 3: 51%) | **3.5%** (12: 7%, 13: 0%) | **25.5%** (22: 0%, 23: 51%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] |
| Client 2 | **1.0%** (4: 1%, 5: 1%) | **4.0%** (14: 8%, 15: 0%) | **4.0%** (24: 8%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **11.0%** (44: 0%, 45: 22%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **31.0%** (36: 37%, 37: 25%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] |
| Client 4 | **31.0%** (8: 60%, 9: 2%) | **33.0%** (18: 66%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) | **1.0%** (38: 0%, 39: 2%) | **4.0%** (48: 4%, 49: 4%) | **0.0%** (58: 0%, 59: 0%) |
