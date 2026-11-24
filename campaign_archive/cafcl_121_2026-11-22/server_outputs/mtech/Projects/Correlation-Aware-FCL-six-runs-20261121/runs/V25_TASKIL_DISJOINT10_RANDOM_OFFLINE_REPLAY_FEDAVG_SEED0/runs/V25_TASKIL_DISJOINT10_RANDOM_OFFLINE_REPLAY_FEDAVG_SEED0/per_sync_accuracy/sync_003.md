# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 57.53% on 30 observed labels.
Validation accuracy: 0.5866666666666667; seen labels never predicted: 4.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.5803571428571429.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **53.0%** (0: 6%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) | **79.5%** (20: 59%, 21: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **65.0%** (12: 50%, 13: 80%) | **50.5%** (22: 1%, 23: 100%) [PENDING upload] |
| Client 2 | **68.0%** (4: 93%, 5: 43%) | **47.0%** (14: 32%, 15: 62%) | **50.0%** (24: 0%, 25: 100%) |
| Client 3 | **50.5%** (6: 1%, 7: 100%) | **81.5%** (16: 79%, 17: 84%) | **50.5%** (26: 1%, 27: 100%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) | **56.0%** (18: 12%, 19: 100%) | **61.5%** (28: 52%, 29: 71%) |
