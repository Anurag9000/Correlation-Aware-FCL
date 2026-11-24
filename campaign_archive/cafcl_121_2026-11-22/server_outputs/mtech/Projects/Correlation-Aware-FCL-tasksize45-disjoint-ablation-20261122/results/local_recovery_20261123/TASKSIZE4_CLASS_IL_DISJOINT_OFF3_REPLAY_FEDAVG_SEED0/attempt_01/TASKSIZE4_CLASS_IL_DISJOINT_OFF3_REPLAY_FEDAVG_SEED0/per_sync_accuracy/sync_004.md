# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 2.08% on 80 observed labels.
Validation accuracy: 0.02225; seen labels never predicted: 74.
Server-accepted classes: 52; pending private classes: 28.
Accepted-only TEST accuracy: 0.03192307692307692.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) [PENDING upload] | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%) [PENDING upload] |
| Client 1 | **7.2%** (4: 29%, 5: 0%, 6: 0%, 7: 0%) | **0.0%** (24: 0%, 25: 0%, 26: 0%, 27: 0%) | **0.0%** (44: 0%, 45: 0%, 46: 0%, 47: 0%) | **0.0%** (64: 0%, 65: 0%, 66: 0%, 67: 0%) [PENDING upload] |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) [PENDING upload] | **0.0%** (68: 0%, 69: 0%, 70: 0%, 71: 0%) [PENDING upload] |
| Client 3 | **0.0%** (12: 0%, 13: 0%, 14: 0%, 15: 0%) | **0.0%** (32: 0%, 33: 0%, 34: 0%, 35: 0%) | **0.0%** (52: 0%, 53: 0%, 54: 0%, 55: 0%) | **0.0%** (72: 0%, 73: 0%, 74: 0%, 75: 0%) |
| Client 4 | **0.0%** (16: 0%, 17: 0%, 18: 0%, 19: 0%) | **19.2%** (36: 0%, 37: 0%, 38: 0%, 39: 77%) | **0.0%** (56: 0%, 57: 0%, 58: 0%, 59: 0%) | **15.0%** (76: 0%, 77: 0%, 78: 60%, 79: 0%) |
