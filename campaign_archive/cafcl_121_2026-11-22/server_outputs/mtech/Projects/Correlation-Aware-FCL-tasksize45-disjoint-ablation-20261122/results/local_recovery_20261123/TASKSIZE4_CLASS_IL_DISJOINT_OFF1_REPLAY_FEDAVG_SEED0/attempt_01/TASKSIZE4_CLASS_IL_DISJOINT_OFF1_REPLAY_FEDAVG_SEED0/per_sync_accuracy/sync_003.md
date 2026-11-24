# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.62% on 60 observed labels.
Validation accuracy: 0.055; seen labels never predicted: 39.
Server-accepted classes: 56; pending private classes: 4.
Accepted-only TEST accuracy: 0.06017857142857143.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **4.8%** (0: 1%, 1: 4%, 2: 13%, 3: 1%) | **12.0%** (20: 0%, 21: 0%, 22: 0%, 23: 48%) | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) |
| Client 1 | **23.2%** (4: 77%, 5: 11%, 6: 0%, 7: 5%) | **0.0%** (24: 0%, 25: 0%, 26: 0%, 27: 0%) | **0.0%** (44: 0%, 45: 0%, 46: 0%, 47: 0%) [PENDING upload] |
| Client 2 | **22.2%** (8: 87%, 9: 0%, 10: 1%, 11: 1%) | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) |
| Client 3 | **19.5%** (12: 54%, 13: 13%, 14: 1%, 15: 10%) | **2.0%** (32: 0%, 33: 8%, 34: 0%, 35: 0%) | **0.0%** (52: 0%, 53: 0%, 54: 0%, 55: 0%) |
| Client 4 | **0.0%** (16: 0%, 17: 0%, 18: 0%, 19: 0%) | **0.5%** (36: 0%, 37: 0%, 38: 0%, 39: 2%) | **0.0%** (56: 0%, 57: 0%, 58: 0%, 59: 0%) |
