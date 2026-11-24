# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.70% on 80 observed labels.
Validation accuracy: 0.137; seen labels never predicted: 30.
Server-accepted classes: 62; pending private classes: 18.
Accepted-only TEST accuracy: 0.1767741935483871.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **16.0%** (0: 16%, 1: 16%) | **0.0%** (10: 0%, 11: 0%) | **4.5%** (20: 9%, 21: 0%) | **23.0%** (30: 43%, 31: 3%) | **9.0%** (40: 0%, 41: 18%) | **1.5%** (50: 0%, 51: 3%) | **4.5%** (60: 5%, 61: 4%) | **0.0%** (70: 0%, 71: 0%) [PENDING upload] |
| Client 1 | **12.0%** (2: 8%, 3: 16%) | **44.0%** (12: 27%, 13: 61%) | **35.0%** (22: 2%, 23: 68%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] | **0.0%** (72: 0%, 73: 0%) [PENDING upload] |
| Client 2 | **13.0%** (4: 19%, 5: 7%) | **39.5%** (14: 37%, 15: 42%) | **18.5%** (24: 34%, 25: 3%) | **4.5%** (34: 8%, 35: 1%) | **5.0%** (44: 9%, 45: 1%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] | **0.0%** (74: 0%, 75: 0%) [PENDING upload] |
| Client 3 | **40.0%** (6: 32%, 7: 48%) | **40.0%** (16: 21%, 17: 59%) | **22.5%** (26: 0%, 27: 45%) | **25.5%** (36: 16%, 37: 35%) | **0.0%** (46: 0%, 47: 0%) | **0.0%** (56: 0%, 57: 0%) | **0.0%** (66: 0%, 67: 0%) | **0.0%** (76: 0%, 77: 0%) |
| Client 4 | **37.0%** (8: 49%, 9: 25%) | **36.5%** (18: 58%, 19: 15%) | **19.0%** (28: 20%, 29: 18%) | **19.5%** (38: 27%, 39: 12%) | **23.0%** (48: 31%, 49: 15%) | **21.5%** (58: 8%, 59: 35%) | **33.5%** (68: 42%, 69: 25%) | **0.0%** (78: 0%, 79: 0%) |
