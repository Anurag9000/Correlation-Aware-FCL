# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.77% on 60 observed labels.
Validation accuracy: 0.2876666666666667; seen labels never predicted: 18.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.4059090909090909.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **55.5%** (0: 59%, 1: 52%) | **21.0%** (10: 34%, 11: 8%) | **60.5%** (20: 68%, 21: 53%) | **47.0%** (30: 67%, 31: 27%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] | **0.0%** (50: 0%, 51: 0%) [PENDING upload] |
| Client 1 | **44.5%** (2: 50%, 3: 39%) | **50.0%** (12: 59%, 13: 41%) | **52.0%** (22: 27%, 23: 77%) | **41.5%** (32: 20%, 33: 63%) | **53.5%** (42: 59%, 43: 48%) | **1.0%** (52: 0%, 53: 2%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **46.5%** (6: 47%, 7: 46%) | **50.5%** (16: 27%, 17: 74%) | **45.0%** (26: 40%, 27: 50%) | **51.0%** (36: 48%, 37: 54%) | **44.0%** (46: 11%, 47: 77%) | **0.0%** (56: 0%, 57: 0%) |
| Client 4 | **54.5%** (8: 60%, 9: 49%) | **41.5%** (18: 44%, 19: 39%) | **43.0%** (28: 53%, 29: 33%) | **26.0%** (38: 22%, 39: 30%) | **63.5%** (48: 71%, 49: 56%) | **1.0%** (58: 1%, 59: 1%) |
