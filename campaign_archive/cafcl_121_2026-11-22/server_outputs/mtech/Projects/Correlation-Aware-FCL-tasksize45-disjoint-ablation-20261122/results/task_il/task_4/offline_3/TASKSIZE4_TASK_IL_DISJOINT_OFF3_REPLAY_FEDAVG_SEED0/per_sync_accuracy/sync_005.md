# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.19% on 100 observed labels.
Validation accuracy: 0.2952; seen labels never predicted: 51.
Server-accepted classes: 68; pending private classes: 32.
Accepted-only TEST accuracy: 0.3114705882352941.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **29.5%** (0: 74%, 1: 0%, 2: 44%, 3: 0%) [PENDING upload] | **25.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **25.0%** (40: 0%, 41: 100%, 42: 0%, 43: 0%) [PENDING upload] | **27.5%** (60: 0%, 61: 11%, 62: 99%, 63: 0%) [PENDING upload] | **25.0%** (80: 0%, 81: 0%, 82: 0%, 83: 100%) [PENDING upload] |
| Client 1 | **30.0%** (4: 100%, 5: 20%, 6: 0%, 7: 0%) | **26.8%** (24: 0%, 25: 9%, 26: 0%, 27: 98%) | **25.0%** (44: 100%, 45: 0%, 46: 0%, 47: 0%) | **18.2%** (64: 0%, 65: 0%, 66: 72%, 67: 1%) [PENDING upload] | **25.5%** (84: 0%, 85: 0%, 86: 4%, 87: 98%) [PENDING upload] |
| Client 2 | **25.0%** (8: 100%, 9: 0%, 10: 0%, 11: 0%) | **49.8%** (28: 68%, 29: 0%, 30: 42%, 31: 89%) | **25.0%** (48: 0%, 49: 0%, 50: 100%, 51: 0%) | **50.5%** (68: 50%, 69: 0%, 70: 52%, 71: 100%) | **40.0%** (88: 76%, 89: 0%, 90: 0%, 91: 84%) |
| Client 3 | **25.2%** (12: 100%, 13: 0%, 14: 0%, 15: 1%) | **27.8%** (32: 11%, 33: 0%, 34: 100%, 35: 0%) | **25.0%** (52: 0%, 53: 0%, 54: 0%, 55: 100%) | **33.2%** (72: 1%, 73: 32%, 74: 100%, 75: 0%) | **24.5%** (92: 98%, 93: 0%, 94: 0%, 95: 0%) [PENDING upload] |
| Client 4 | **25.0%** (16: 0%, 17: 0%, 18: 0%, 19: 100%) | **32.2%** (36: 0%, 37: 0%, 38: 34%, 39: 95%) | **38.5%** (56: 90%, 57: 17%, 58: 47%, 59: 0%) | **25.5%** (76: 0%, 77: 0%, 78: 100%, 79: 2%) | **25.0%** (96: 0%, 97: 0%, 98: 0%, 99: 100%) |
