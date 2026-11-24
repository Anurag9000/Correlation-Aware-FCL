# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.45% on 38 observed labels.
Validation accuracy: 0.8121052631578948; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 0.
Accepted-only TEST accuracy: 0.8044736842105263.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **86.5%** (0: 79%, 1: 94%) | **68.0%** (10: 90%, 11: 46%) | **86.5%** (0: 79%, 1: 94%) | **91.0%** (30: 95%, 31: 87%) |
| Client 1 | **82.5%** (2: 68%, 3: 97%) | **63.5%** (12: 91%, 13: 36%) | **85.5%** (22: 73%, 23: 98%) | **77.5%** (32: 64%, 33: 91%) |
| Client 2 | **84.5%** (4: 93%, 5: 76%) | **71.5%** (14: 54%, 15: 89%) | **86.0%** (24: 83%, 25: 89%) | **76.5%** (34: 94%, 35: 59%) |
| Client 3 | **77.5%** (6: 76%, 7: 79%) | **90.5%** (16: 84%, 17: 97%) | **83.5%** (26: 70%, 27: 97%) | **80.5%** (36: 70%, 37: 91%) |
| Client 4 | **83.0%** (8: 85%, 9: 81%) | **81.5%** (18: 81%, 19: 82%) | **81.0%** (28: 73%, 29: 89%) | **78.0%** (38: 85%, 39: 71%) |
