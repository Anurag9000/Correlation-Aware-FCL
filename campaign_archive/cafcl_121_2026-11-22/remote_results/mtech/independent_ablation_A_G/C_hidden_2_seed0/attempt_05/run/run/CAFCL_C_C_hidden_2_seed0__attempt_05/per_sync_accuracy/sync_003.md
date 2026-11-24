# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.10% on 30 observed labels.
Validation accuracy: 0.6986666666666667; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.7021428571428572.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **56.5%** (8: 100%, 9: 13%) | **64.0%** (14: 31%, 15: 97%) | **71.5%** (26: 53%, 27: 90%) |
| Client 1 | **51.5%** (6: 3%, 7: 100%) | **61.5%** (18: 79%, 19: 44%) | **53.0%** (28: 82%, 29: 24%) |
| Client 2 | **65.0%** (4: 98%, 5: 32%) | **86.0%** (16: 73%, 17: 99%) | **82.0%** (22: 95%, 23: 69%) |
| Client 3 | **80.5%** (0: 84%, 1: 77%) | **61.0%** (10: 94%, 11: 28%) | **95.5%** (20: 93%, 21: 98%) |
| Client 4 | **83.5%** (2: 73%, 3: 94%) | **71.5%** (12: 69%, 13: 74%) | **53.5%** (24: 99%, 25: 8%) [PENDING upload] |
