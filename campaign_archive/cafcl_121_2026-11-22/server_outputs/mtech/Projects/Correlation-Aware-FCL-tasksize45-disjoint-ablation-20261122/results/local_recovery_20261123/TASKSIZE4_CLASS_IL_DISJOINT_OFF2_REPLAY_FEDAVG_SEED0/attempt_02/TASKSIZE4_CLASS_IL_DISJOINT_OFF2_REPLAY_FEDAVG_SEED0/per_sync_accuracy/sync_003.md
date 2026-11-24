# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.82% on 60 observed labels.
Validation accuracy: 0.18466666666666667; seen labels never predicted: 25.
Server-accepted classes: 40; pending private classes: 20.
Accepted-only TEST accuracy: 0.29725.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **46.5%** (0: 63%, 1: 62%, 2: 32%, 3: 29%) | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) [PENDING upload] |
| Client 1 | **47.5%** (4: 54%, 5: 49%, 6: 42%, 7: 45%) | **37.2%** (24: 68%, 25: 5%, 26: 37%, 27: 39%) | **0.8%** (44: 0%, 45: 0%, 46: 0%, 47: 3%) |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) [PENDING upload] |
| Client 3 | **43.0%** (12: 62%, 13: 53%, 14: 20%, 15: 37%) | **28.7%** (32: 13%, 33: 61%, 34: 17%, 35: 24%) | **0.5%** (52: 1%, 53: 1%, 54: 0%, 55: 0%) |
| Client 4 | **42.8%** (16: 37%, 17: 62%, 18: 50%, 19: 22%) | **49.0%** (36: 44%, 37: 48%, 38: 37%, 39: 67%) | **1.2%** (56: 0%, 57: 0%, 58: 0%, 59: 5%) |
