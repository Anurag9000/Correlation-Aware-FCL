# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.65% on 60 observed labels.
Validation accuracy: 0.18433333333333332; seen labels never predicted: 27.
Server-accepted classes: 40; pending private classes: 20.
Accepted-only TEST accuracy: 0.29475.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **44.2%** (0: 50%, 1: 57%, 2: 46%, 3: 24%) | **0.0%** (20: 0%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%) [PENDING upload] |
| Client 1 | **48.8%** (4: 58%, 5: 48%, 6: 42%, 7: 47%) | **38.0%** (24: 68%, 25: 6%, 26: 31%, 27: 47%) | **0.2%** (44: 0%, 45: 0%, 46: 0%, 47: 1%) |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] | **0.0%** (28: 0%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] | **0.0%** (48: 0%, 49: 0%, 50: 0%, 51: 0%) [PENDING upload] |
| Client 3 | **36.5%** (12: 49%, 13: 42%, 14: 19%, 15: 36%) | **34.8%** (32: 15%, 33: 63%, 34: 33%, 35: 28%) | **2.0%** (52: 3%, 53: 1%, 54: 4%, 55: 0%) |
| Client 4 | **44.8%** (16: 31%, 17: 67%, 18: 53%, 19: 28%) | **44.2%** (36: 37%, 37: 39%, 38: 34%, 39: 67%) | **1.2%** (56: 0%, 57: 0%, 58: 0%, 59: 5%) |
