# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.65% on 100 observed labels.
Validation accuracy: 0.2154; seen labels never predicted: 64.
Server-accepted classes: 65; pending private classes: 35.
Accepted-only TEST accuracy: 0.22507692307692306.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **19.0%** (0: 0%, 1: 42%, 2: 0%, 3: 0%, 4: 53%) [PENDING upload] | **21.2%** (25: 0%, 26: 87%, 27: 19%, 28: 0%, 29: 0%) [PENDING upload] | **20.0%** (50: 0%, 51: 0%, 52: 100%, 53: 0%, 54: 0%) [PENDING upload] | **20.0%** (75: 0%, 76: 0%, 77: 0%, 78: 100%, 79: 0%) [PENDING upload] |
| Client 1 | **24.4%** (5: 31%, 6: 0%, 7: 0%, 8: 91%, 9: 0%) | **20.0%** (30: 0%, 31: 0%, 32: 0%, 33: 100%, 34: 0%) | **20.8%** (55: 100%, 56: 0%, 57: 0%, 58: 4%, 59: 0%) | **20.0%** (80: 0%, 81: 0%, 82: 100%, 83: 0%, 84: 0%) [PENDING upload] |
| Client 2 | **22.4%** (10: 12%, 11: 0%, 12: 0%, 13: 100%, 14: 0%) | **22.6%** (35: 0%, 36: 0%, 37: 0%, 38: 20%, 39: 93%) | **20.0%** (60: 0%, 61: 100%, 62: 0%, 63: 0%, 64: 0%) [PENDING upload] | **20.2%** (85: 7%, 86: 0%, 87: 0%, 88: 0%, 89: 94%) [PENDING upload] |
| Client 3 | **20.0%** (15: 0%, 16: 0%, 17: 0%, 18: 100%, 19: 0%) | **24.2%** (40: 0%, 41: 38%, 42: 45%, 43: 0%, 44: 38%) | **31.4%** (65: 82%, 66: 0%, 67: 75%, 68: 0%, 69: 0%) | **23.0%** (90: 0%, 91: 100%, 92: 0%, 93: 15%, 94: 0%) |
| Client 4 | **22.4%** (20: 0%, 21: 0%, 22: 12%, 23: 100%, 24: 0%) | **21.4%** (45: 100%, 46: 0%, 47: 7%, 48: 0%, 49: 0%) | **20.0%** (70: 0%, 71: 0%, 72: 0%, 73: 100%, 74: 0%) | **20.0%** (95: 0%, 96: 0%, 97: 0%, 98: 0%, 99: 100%) |
