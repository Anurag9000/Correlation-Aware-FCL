# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.50% on 80 observed labels.
Validation accuracy: 0.11025; seen labels never predicted: 41.
Server-accepted classes: 60; pending private classes: 20.
Accepted-only TEST accuracy: 0.15333333333333332.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **34.0%** (0: 26%, 1: 31%, 2: 58%, 3: 21%) | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%) |
| Client 1 | **46.8%** (4: 74%, 5: 19%, 6: 41%, 7: 53%) | **14.8%** (24: 6%, 25: 7%, 26: 27%, 27: 19%) | **4.8%** (44: 4%, 45: 7%, 46: 1%, 47: 7%) | **0.0%** (64: 0%, 65: 0%, 66: 0%, 67: 0%) [PENDING upload] |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) [PENDING upload] | **0.0%** (68: 0%, 69: 0%, 70: 0%, 71: 0%) [PENDING upload] |
| Client 3 | **26.8%** (12: 48%, 13: 37%, 14: 17%, 15: 5%) | **23.0%** (32: 2%, 33: 70%, 34: 16%, 35: 4%) | **6.8%** (52: 11%, 53: 5%, 54: 11%, 55: 0%) | **0.0%** (72: 0%, 73: 0%, 74: 0%, 75: 0%) |
| Client 4 | **33.2%** (16: 28%, 17: 34%, 18: 34%, 19: 37%) | **34.8%** (36: 4%, 37: 28%, 38: 23%, 39: 84%) | **5.3%** (56: 2%, 57: 5%, 58: 0%, 59: 14%) | **0.0%** (76: 0%, 77: 0%, 78: 0%, 79: 0%) |
