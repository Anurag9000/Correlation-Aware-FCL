# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.79% on 75 observed labels.
Validation accuracy: 0.5656; seen labels never predicted: 2.
Server-accepted classes: 50; pending private classes: 25.
Accepted-only TEST accuracy: 0.7584.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **74.6%** (0: 82%, 1: 73%, 2: 84%, 3: 67%, 4: 67%) | **16.4%** (25: 12%, 26: 46%, 27: 0%, 28: 24%, 29: 0%) [PENDING upload] | **24.2%** (50: 4%, 51: 8%, 52: 60%, 53: 0%, 54: 49%) [PENDING upload] |
| Client 1 | **76.4%** (5: 78%, 6: 65%, 7: 70%, 8: 86%, 9: 83%) | **77.2%** (30: 86%, 31: 77%, 32: 59%, 33: 80%, 34: 84%) | **66.8%** (55: 94%, 56: 50%, 57: 65%, 58: 40%, 59: 85%) |
| Client 2 | **16.8%** (10: 16%, 11: 50%, 12: 1%, 13: 0%, 14: 17%) [PENDING upload] | **21.8%** (35: 15%, 36: 57%, 37: 32%, 38: 0%, 39: 5%) [PENDING upload] | **14.2%** (60: 0%, 61: 40%, 62: 29%, 63: 0%, 64: 2%) [PENDING upload] |
| Client 3 | **71.4%** (15: 42%, 16: 74%, 17: 90%, 18: 80%, 19: 71%) | **70.8%** (40: 79%, 41: 66%, 42: 71%, 43: 84%, 44: 54%) | **71.2%** (65: 80%, 66: 65%, 67: 83%, 68: 76%, 69: 52%) |
| Client 4 | **87.2%** (20: 80%, 21: 97%, 22: 75%, 23: 93%, 24: 91%) | **81.6%** (45: 56%, 46: 90%, 47: 89%, 48: 82%, 49: 91%) | **81.2%** (70: 92%, 71: 82%, 72: 92%, 73: 74%, 74: 66%) |
