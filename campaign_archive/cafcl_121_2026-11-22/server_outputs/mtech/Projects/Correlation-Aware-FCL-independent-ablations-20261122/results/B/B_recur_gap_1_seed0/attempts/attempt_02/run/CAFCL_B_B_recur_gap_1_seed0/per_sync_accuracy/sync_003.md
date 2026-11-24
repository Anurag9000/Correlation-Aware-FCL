# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.46% on 28 observed labels.
Validation accuracy: 0.7214285714285714; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.7246428571428571.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **80.0%** (0: 62%, 1: 98%) | **52.5%** (10: 99%, 11: 6%) | **80.0%** (0: 62%, 1: 98%) |
| Client 1 | **69.5%** (2: 40%, 3: 99%) | **61.5%** (12: 93%, 13: 30%) | **81.0%** (22: 65%, 23: 97%) |
| Client 2 | **79.0%** (4: 97%, 5: 61%) | **73.0%** (14: 54%, 15: 92%) | **81.0%** (24: 77%, 25: 85%) |
| Client 3 | **62.5%** (6: 34%, 7: 91%) | **84.0%** (16: 69%, 17: 99%) | **68.0%** (26: 37%, 27: 99%) |
| Client 4 | **73.0%** (8: 96%, 9: 50%) | **72.0%** (18: 81%, 19: 63%) | **77.5%** (28: 65%, 29: 90%) |
