# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.97% on 30 observed labels.
Validation accuracy: 0.7406666666666667; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.7396666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **77.0%** (0: 59%, 1: 95%) | **53.0%** (10: 95%, 11: 11%) | **83.0%** (20: 66%, 21: 100%) |
| Client 1 | **71.0%** (2: 42%, 3: 100%) | **67.5%** (12: 87%, 13: 48%) | **80.0%** (22: 68%, 23: 92%) |
| Client 2 | **79.0%** (4: 95%, 5: 63%) | **66.5%** (14: 34%, 15: 99%) | **81.5%** (24: 69%, 25: 94%) |
| Client 3 | **67.0%** (6: 35%, 7: 99%) | **86.5%** (16: 78%, 17: 95%) | **68.5%** (26: 38%, 27: 99%) |
| Client 4 | **77.5%** (8: 98%, 9: 57%) | **81.0%** (18: 67%, 19: 95%) | **70.5%** (28: 59%, 29: 82%) |
