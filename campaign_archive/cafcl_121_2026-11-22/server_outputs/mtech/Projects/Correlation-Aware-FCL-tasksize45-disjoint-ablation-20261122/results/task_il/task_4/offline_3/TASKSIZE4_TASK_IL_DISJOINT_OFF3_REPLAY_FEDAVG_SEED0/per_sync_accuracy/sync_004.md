# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.88% on 80 observed labels.
Validation accuracy: 0.2635; seen labels never predicted: 45.
Server-accepted classes: 52; pending private classes: 28.
Accepted-only TEST accuracy: 0.2876923076923077.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **25.2%** (0: 100%, 1: 0%, 2: 1%, 3: 0%) [PENDING upload] | **25.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **25.0%** (40: 0%, 41: 100%, 42: 0%, 43: 0%) [PENDING upload] | **25.2%** (60: 0%, 61: 1%, 62: 100%, 63: 0%) [PENDING upload] |
| Client 1 | **27.8%** (4: 100%, 5: 11%, 6: 0%, 7: 0%) | **25.0%** (24: 0%, 25: 0%, 26: 0%, 27: 100%) | **25.0%** (44: 100%, 45: 0%, 46: 0%, 47: 0%) | **13.0%** (64: 0%, 65: 0%, 66: 33%, 67: 19%) [PENDING upload] |
| Client 2 | **25.2%** (8: 100%, 9: 0%, 10: 0%, 11: 1%) | **28.7%** (28: 12%, 29: 0%, 30: 3%, 31: 100%) | **25.0%** (48: 0%, 49: 0%, 50: 0%, 51: 100%) [PENDING upload] | **25.0%** (68: 0%, 69: 0%, 70: 100%, 71: 0%) [PENDING upload] |
| Client 3 | **39.2%** (12: 87%, 13: 13%, 14: 0%, 15: 57%) | **25.2%** (32: 1%, 33: 0%, 34: 100%, 35: 0%) | **25.0%** (52: 0%, 53: 0%, 54: 0%, 55: 100%) | **32.5%** (72: 0%, 73: 30%, 74: 100%, 75: 0%) |
| Client 4 | **25.0%** (16: 0%, 17: 0%, 18: 0%, 19: 100%) | **40.2%** (36: 0%, 37: 0%, 38: 90%, 39: 71%) | **30.0%** (56: 24%, 57: 0%, 58: 2%, 59: 94%) | **25.0%** (76: 0%, 77: 0%, 78: 100%, 79: 0%) |
