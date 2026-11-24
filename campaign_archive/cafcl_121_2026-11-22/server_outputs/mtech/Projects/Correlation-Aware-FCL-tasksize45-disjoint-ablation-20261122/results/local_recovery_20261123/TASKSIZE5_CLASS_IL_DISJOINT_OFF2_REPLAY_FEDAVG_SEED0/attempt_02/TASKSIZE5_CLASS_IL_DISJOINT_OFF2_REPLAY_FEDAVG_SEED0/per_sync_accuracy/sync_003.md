# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.73% on 75 observed labels.
Validation accuracy: 0.23866666666666667; seen labels never predicted: 39.
Server-accepted classes: 50; pending private classes: 25.
Accepted-only TEST accuracy: 0.356.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **53.6%** (0: 74%, 1: 64%, 2: 56%, 3: 26%, 4: 48%) | **0.0%** (25: 0%, 26: 0%, 27: 0%, 28: 0%, 29: 0%) [PENDING upload] | **0.0%** (50: 0%, 51: 0%, 52: 0%, 53: 0%, 54: 0%) [PENDING upload] |
| Client 1 | **46.6%** (5: 35%, 6: 44%, 7: 29%, 8: 65%, 9: 60%) | **54.4%** (30: 81%, 31: 54%, 32: 37%, 33: 61%, 34: 39%) | **0.0%** (55: 0%, 56: 0%, 57: 0%, 58: 0%, 59: 0%) |
| Client 2 | **0.0%** (10: 0%, 11: 0%, 12: 0%, 13: 0%, 14: 0%) [PENDING upload] | **0.0%** (35: 0%, 36: 0%, 37: 0%, 38: 0%, 39: 0%) [PENDING upload] | **0.0%** (60: 0%, 61: 0%, 62: 0%, 63: 0%, 64: 0%) [PENDING upload] |
| Client 3 | **36.8%** (15: 19%, 16: 28%, 17: 86%, 18: 41%, 19: 10%) | **46.2%** (40: 23%, 41: 55%, 42: 62%, 43: 47%, 44: 44%) | **0.0%** (65: 0%, 66: 0%, 67: 0%, 68: 0%, 69: 0%) |
| Client 4 | **68.6%** (20: 69%, 21: 82%, 22: 46%, 23: 83%, 24: 63%) | **49.8%** (45: 27%, 46: 36%, 47: 83%, 48: 40%, 49: 63%) | **0.0%** (70: 0%, 71: 0%, 72: 0%, 73: 0%, 74: 0%) |
