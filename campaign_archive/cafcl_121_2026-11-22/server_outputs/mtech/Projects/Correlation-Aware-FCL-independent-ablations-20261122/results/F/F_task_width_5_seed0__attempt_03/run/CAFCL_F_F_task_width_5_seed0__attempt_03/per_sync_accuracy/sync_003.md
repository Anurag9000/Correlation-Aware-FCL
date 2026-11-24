# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.95% on 75 observed labels.
Validation accuracy: 0.7384; seen labels never predicted: 0.
Server-accepted classes: 75; pending private classes: 0.
Accepted-only TEST accuracy: 0.7394666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **72.8%** (0: 81%, 1: 70%, 2: 82%, 3: 50%, 4: 81%) | **67.6%** (25: 50%, 26: 70%, 27: 81%, 28: 78%, 29: 59%) | **74.0%** (50: 97%, 51: 35%, 52: 98%, 53: 90%, 54: 50%) |
| Client 1 | **76.6%** (5: 77%, 6: 68%, 7: 79%, 8: 84%, 9: 75%) | **77.2%** (30: 88%, 31: 83%, 32: 58%, 33: 77%, 34: 80%) | **67.2%** (55: 93%, 56: 56%, 57: 57%, 58: 70%, 59: 60%) |
| Client 2 | **69.8%** (10: 53%, 11: 78%, 12: 81%, 13: 59%, 14: 78%) | **68.4%** (35: 44%, 36: 74%, 37: 73%, 38: 87%, 39: 64%) | **72.6%** (60: 94%, 61: 66%, 62: 90%, 63: 96%, 64: 17%) |
| Client 3 | **70.8%** (15: 75%, 16: 75%, 17: 87%, 18: 84%, 19: 33%) | **73.0%** (40: 64%, 41: 77%, 42: 77%, 43: 81%, 44: 66%) | **70.4%** (65: 74%, 66: 75%, 67: 77%, 68: 78%, 69: 48%) |
| Client 4 | **86.6%** (20: 79%, 21: 97%, 22: 70%, 23: 91%, 24: 96%) | **83.4%** (45: 74%, 46: 82%, 47: 83%, 48: 87%, 49: 91%) | **78.8%** (70: 86%, 71: 86%, 72: 84%, 73: 65%, 74: 73%) |
