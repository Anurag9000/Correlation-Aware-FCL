# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.90% on 40 observed labels.
Validation accuracy: 0.7645; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.769.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **81.5%** (0: 73%, 1: 90%) | **63.0%** (10: 76%, 11: 50%) | **92.5%** (20: 89%, 21: 96%) | **85.5%** (30: 92%, 31: 79%) |
| Client 1 | **86.5%** (2: 78%, 3: 95%) | **62.0%** (12: 94%, 13: 30%) | **80.5%** (22: 69%, 23: 92%) | **64.0%** (32: 31%, 33: 97%) |
| Client 2 | **80.0%** (4: 82%, 5: 78%) | **71.0%** (14: 52%, 15: 90%) | **82.5%** (24: 81%, 25: 84%) | **76.0%** (34: 69%, 35: 83%) |
| Client 3 | **73.0%** (6: 56%, 7: 90%) | **89.5%** (16: 80%, 17: 99%) | **68.5%** (26: 38%, 27: 99%) | **79.5%** (36: 61%, 37: 98%) |
| Client 4 | **76.0%** (8: 89%, 9: 63%) | **75.5%** (18: 90%, 19: 61%) | **77.5%** (28: 68%, 29: 87%) | **73.5%** (38: 83%, 39: 64%) |
