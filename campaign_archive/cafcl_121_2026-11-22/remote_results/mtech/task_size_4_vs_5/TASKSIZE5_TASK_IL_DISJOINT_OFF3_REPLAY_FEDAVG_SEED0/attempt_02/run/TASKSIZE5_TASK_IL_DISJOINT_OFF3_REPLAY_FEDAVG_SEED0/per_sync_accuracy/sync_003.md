# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.20% on 75 observed labels.
Validation accuracy: 0.22266666666666668; seen labels never predicted: 46.
Server-accepted classes: 50; pending private classes: 25.
Accepted-only TEST accuracy: 0.2334.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **20.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%, 4: 100%) [PENDING upload] | **21.6%** (25: 26%, 26: 1%, 27: 81%, 28: 0%, 29: 0%) [PENDING upload] | **18.0%** (50: 75%, 51: 0%, 52: 15%, 53: 0%, 54: 0%) [PENDING upload] |
| Client 1 | **20.6%** (5: 4%, 6: 0%, 7: 0%, 8: 99%, 9: 0%) | **30.2%** (30: 0%, 31: 0%, 32: 91%, 33: 60%, 34: 0%) | **21.6%** (55: 0%, 56: 0%, 57: 0%, 58: 96%, 59: 12%) |
| Client 2 | **24.4%** (10: 98%, 11: 0%, 12: 24%, 13: 0%, 14: 0%) | **20.2%** (35: 6%, 36: 0%, 37: 82%, 38: 0%, 39: 13%) | **20.0%** (60: 100%, 61: 0%, 62: 0%, 63: 0%, 64: 0%) [PENDING upload] |
| Client 3 | **21.2%** (15: 0%, 16: 0%, 17: 100%, 18: 0%, 19: 6%) | **20.0%** (40: 0%, 41: 0%, 42: 99%, 43: 1%, 44: 0%) | **20.0%** (65: 0%, 66: 100%, 67: 0%, 68: 0%, 69: 0%) |
| Client 4 | **35.2%** (20: 0%, 21: 0%, 22: 82%, 23: 94%, 24: 0%) | **20.0%** (45: 100%, 46: 0%, 47: 0%, 48: 0%, 49: 0%) | **20.0%** (70: 0%, 71: 0%, 72: 100%, 73: 0%, 74: 0%) [PENDING upload] |
