# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.00% on 80 observed labels.
Validation accuracy: 0.12075; seen labels never predicted: 41.
Server-accepted classes: 60; pending private classes: 20.
Accepted-only TEST accuracy: 0.16.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **29.5%** (0: 32%, 1: 36%, 2: 39%, 3: 11%) | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%) |
| Client 1 | **53.0%** (4: 83%, 5: 41%, 6: 36%, 7: 52%) | **21.5%** (24: 12%, 25: 4%, 26: 31%, 27: 39%) | **4.5%** (44: 4%, 45: 4%, 46: 1%, 47: 9%) | **0.0%** (64: 0%, 65: 0%, 66: 0%, 67: 0%) [PENDING upload] |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) [PENDING upload] | **0.0%** (68: 0%, 69: 0%, 70: 0%, 71: 0%) [PENDING upload] |
| Client 3 | **33.8%** (12: 59%, 13: 54%, 14: 13%, 15: 9%) | **18.0%** (32: 0%, 33: 55%, 34: 13%, 35: 4%) | **6.2%** (52: 16%, 53: 5%, 54: 4%, 55: 0%) | **0.0%** (72: 0%, 73: 0%, 74: 0%, 75: 0%) |
| Client 4 | **29.2%** (16: 20%, 17: 49%, 18: 34%, 19: 14%) | **35.5%** (36: 14%, 37: 23%, 38: 30%, 39: 75%) | **8.8%** (56: 3%, 57: 7%, 58: 0%, 59: 25%) | **0.0%** (76: 0%, 77: 0%, 78: 0%, 79: 0%) |
