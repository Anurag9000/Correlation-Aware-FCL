# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.33% on 30 observed labels.
Validation accuracy: 0.7013333333333334; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.6933333333333334.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **67.5%** (0: 41%, 1: 94%) | **54.0%** (10: 79%, 11: 29%) | **91.0%** (20: 84%, 21: 98%) |
| Client 1 | **81.0%** (2: 64%, 3: 98%) | **61.0%** (12: 92%, 13: 30%) | **71.0%** (22: 49%, 23: 93%) |
| Client 2 | **71.5%** (4: 87%, 5: 56%) | **66.5%** (14: 38%, 15: 95%) | **50.5%** (24: 1%, 25: 100%) |
| Client 3 | **66.5%** (6: 49%, 7: 84%) | **81.0%** (16: 62%, 17: 100%) | **71.0%** (26: 47%, 27: 95%) |
| Client 4 | **61.5%** (8: 97%, 9: 26%) | **75.5%** (18: 86%, 19: 65%) | **70.5%** (28: 85%, 29: 56%) |
