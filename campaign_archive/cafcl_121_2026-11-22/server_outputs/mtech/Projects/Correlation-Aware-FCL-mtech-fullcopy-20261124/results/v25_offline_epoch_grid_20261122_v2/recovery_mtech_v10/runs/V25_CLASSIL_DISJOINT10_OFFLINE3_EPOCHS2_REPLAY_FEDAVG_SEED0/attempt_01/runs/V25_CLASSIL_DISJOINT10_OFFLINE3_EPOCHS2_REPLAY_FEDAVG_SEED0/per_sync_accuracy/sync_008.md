# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.26% on 80 observed labels.
Validation accuracy: 0.0685; seen labels never predicted: 44.
Server-accepted classes: 62; pending private classes: 18.
Accepted-only TEST accuracy: 0.08080645161290323.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **2.0%** (0: 4%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **6.5%** (30: 13%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) | **0.0%** (50: 0%, 51: 0%) | **31.0%** (60: 6%, 61: 56%) | **0.0%** (70: 0%, 71: 0%) [PENDING upload] |
| Client 1 | **6.0%** (2: 7%, 3: 5%) | **10.5%** (12: 1%, 13: 20%) | **34.5%** (22: 0%, 23: 69%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] | **0.0%** (72: 0%, 73: 0%) [PENDING upload] |
| Client 2 | **11.0%** (4: 22%, 5: 0%) | **13.0%** (14: 14%, 15: 12%) | **0.5%** (24: 1%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **5.0%** (44: 0%, 45: 10%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] | **0.0%** (74: 0%, 75: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **31.0%** (16: 0%, 17: 62%) | **0.0%** (26: 0%, 27: 0%) | **0.0%** (36: 0%, 37: 0%) | **0.0%** (46: 0%, 47: 0%) | **0.0%** (56: 0%, 57: 0%) | **0.0%** (66: 0%, 67: 0%) | **0.0%** (76: 0%, 77: 0%) |
| Client 4 | **10.5%** (8: 0%, 9: 21%) | **17.5%** (18: 18%, 19: 17%) | **2.0%** (28: 0%, 29: 4%) | **29.5%** (38: 59%, 39: 0%) | **9.0%** (48: 0%, 49: 18%) | **8.0%** (58: 16%, 59: 0%) | **21.5%** (68: 33%, 69: 10%) | **1.5%** (78: 3%, 79: 0%) |
