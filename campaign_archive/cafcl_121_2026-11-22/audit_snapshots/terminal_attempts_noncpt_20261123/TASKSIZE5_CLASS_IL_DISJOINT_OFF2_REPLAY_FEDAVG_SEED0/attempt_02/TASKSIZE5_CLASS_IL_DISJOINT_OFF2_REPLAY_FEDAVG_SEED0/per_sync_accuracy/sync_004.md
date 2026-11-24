# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.51% on 100 observed labels.
Validation accuracy: 0.1382; seen labels never predicted: 53.
Server-accepted classes: 75; pending private classes: 25.
Accepted-only TEST accuracy: 0.18013333333333334.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **43.4%** (0: 47%, 1: 59%, 2: 40%, 3: 18%, 4: 53%) | **0.0%** (25: 0%, 26: 0%, 27: 0%, 28: 0%, 29: 0%) | **0.0%** (50: 0%, 51: 0%, 52: 0%, 53: 0%, 54: 0%) | **0.0%** (75: 0%, 76: 0%, 77: 0%, 78: 0%, 79: 0%) |
| Client 1 | **41.2%** (5: 23%, 6: 26%, 7: 27%, 8: 81%, 9: 49%) | **40.2%** (30: 83%, 31: 18%, 32: 14%, 33: 76%, 34: 10%) | **3.8%** (55: 0%, 56: 1%, 57: 2%, 58: 5%, 59: 11%) | **0.0%** (80: 0%, 81: 0%, 82: 0%, 83: 0%, 84: 0%) [PENDING upload] |
| Client 2 | **0.0%** (10: 0%, 11: 0%, 12: 0%, 13: 0%, 14: 0%) [PENDING upload] | **0.0%** (35: 0%, 36: 0%, 37: 0%, 38: 0%, 39: 0%) [PENDING upload] | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%, 64: 0%) [PENDING upload] | **0.0%** (85: 0%, 86: 0%, 87: 0%, 88: 0%, 89: 0%) [PENDING upload] |
| Client 3 | **14.8%** (15: 3%, 16: 12%, 17: 45%, 18: 12%, 19: 2%) | **31.8%** (40: 6%, 41: 35%, 42: 73%, 43: 7%, 44: 38%) | **2.6%** (65: 0%, 66: 0%, 67: 11%, 68: 2%, 69: 0%) | **0.0%** (90: 0%, 91: 0%, 92: 0%, 93: 0%, 94: 0%) |
| Client 4 | **58.8%** (20: 66%, 21: 82%, 22: 52%, 23: 83%, 24: 11%) | **30.4%** (45: 22%, 46: 10%, 47: 88%, 48: 10%, 49: 22%) | **3.2%** (70: 8%, 71: 0%, 72: 3%, 73: 4%, 74: 1%) | **0.0%** (95: 0%, 96: 0%, 97: 0%, 98: 0%, 99: 0%) |
