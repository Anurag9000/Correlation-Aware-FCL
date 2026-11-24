# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.59% on 80 observed labels.
Validation accuracy: 0.114; seen labels never predicted: 35.
Server-accepted classes: 76; pending private classes: 4.
Accepted-only TEST accuracy: 0.12197368421052632.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **14.8%** (0: 3%, 1: 22%, 2: 30%, 3: 4%) | **32.0%** (20: 36%, 21: 6%, 22: 11%, 23: 75%) | **1.2%** (40: 0%, 41: 0%, 42: 5%, 43: 0%) | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%) [PENDING upload] |
| Client 1 | **25.8%** (4: 80%, 5: 5%, 6: 1%, 7: 17%) | **13.0%** (24: 9%, 25: 8%, 26: 12%, 27: 23%) | **0.0%** (44: 0%, 45: 0%, 46: 0%, 47: 0%) | **0.0%** (64: 0%, 65: 0%, 66: 0%, 67: 0%) |
| Client 2 | **33.5%** (8: 84%, 9: 22%, 10: 10%, 11: 18%) | **4.0%** (28: 2%, 29: 0%, 30: 14%, 31: 0%) | **0.2%** (48: 0%, 49: 1%, 50: 0%, 51: 0%) | **0.0%** (68: 0%, 69: 0%, 70: 0%, 71: 0%) |
| Client 3 | **31.5%** (12: 55%, 13: 41%, 14: 7%, 15: 23%) | **24.2%** (32: 11%, 33: 74%, 34: 12%, 35: 0%) | **11.2%** (52: 37%, 53: 0%, 54: 8%, 55: 0%) | **0.0%** (72: 0%, 73: 0%, 74: 0%, 75: 0%) |
| Client 4 | **10.2%** (16: 4%, 17: 30%, 18: 0%, 19: 7%) | **28.0%** (36: 7%, 37: 17%, 38: 24%, 39: 64%) | **2.0%** (56: 1%, 57: 2%, 58: 0%, 59: 5%) | **0.0%** (76: 0%, 77: 0%, 78: 0%, 79: 0%) |
