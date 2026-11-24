# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 42.47% on 30 observed labels.
Validation accuracy: 0.4146666666666667; seen labels never predicted: 3.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.4246666666666667.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **62.0%** (0: 65%, 1: 59%) | **54.5%** (6: 60%, 7: 49%) | **57.0%** (12: 56%, 13: 58%) | **35.0%** (18: 35%, 19: 35%) | **0.0%** (24: 0%, 25: 0%) |
| Client 1 | **39.5%** (2: 45%, 3: 34%) | **63.5%** (8: 71%, 9: 56%) | **50.0%** (14: 43%, 15: 57%) | **69.0%** (20: 72%, 21: 66%) | **2.0%** (26: 2%, 27: 2%) |
| Client 2 | **40.5%** (4: 39%, 5: 42%) | **40.5%** (10: 41%, 11: 40%) | **59.5%** (16: 42%, 17: 77%) | **64.0%** (22: 39%, 23: 89%) | **0.0%** (28: 0%, 29: 0%) |
