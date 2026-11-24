# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 7.11% on 70 observed labels.
Validation accuracy: 0.07714285714285714; seen labels never predicted: 43.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.09576923076923077.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **6.0%** (0: 12%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **9.5%** (30: 17%, 31: 2%) | **0.0%** (40: 0%, 41: 0%) | **0.0%** (50: 0%, 51: 0%) | **57.5%** (60: 51%, 61: 64%) |
| Client 1 | **7.5%** (2: 10%, 3: 5%) | **2.5%** (12: 0%, 13: 5%) | **31.0%** (22: 0%, 23: 62%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] |
| Client 2 | **2.5%** (4: 5%, 5: 0%) | **7.5%** (14: 9%, 15: 6%) | **1.0%** (24: 2%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **5.5%** (44: 0%, 45: 11%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **13.5%** (16: 0%, 17: 27%) | **0.0%** (26: 0%, 27: 0%) | **1.0%** (36: 0%, 37: 2%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] | **0.0%** (66: 0%, 67: 0%) [PENDING upload] |
| Client 4 | **3.0%** (8: 0%, 9: 6%) | **9.5%** (18: 9%, 19: 10%) | **0.0%** (28: 0%, 29: 0%) | **26.5%** (38: 53%, 39: 0%) | **13.5%** (48: 0%, 49: 27%) | **13.0%** (58: 26%, 59: 0%) | **38.5%** (68: 56%, 69: 21%) |
