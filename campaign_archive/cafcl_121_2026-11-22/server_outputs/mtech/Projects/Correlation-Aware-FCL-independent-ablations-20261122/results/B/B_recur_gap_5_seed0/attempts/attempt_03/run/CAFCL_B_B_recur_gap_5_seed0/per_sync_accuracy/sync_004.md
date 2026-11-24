# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.45% on 40 observed labels.
Validation accuracy: 0.8275; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.8145.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **87.5%** (0: 93%, 1: 82%) | **73.5%** (10: 88%, 11: 59%) | **92.5%** (20: 85%, 21: 100%) | **92.0%** (30: 94%, 31: 90%) |
| Client 1 | **85.5%** (2: 73%, 3: 98%) | **68.0%** (12: 86%, 13: 50%) | **88.5%** (22: 82%, 23: 95%) | **78.5%** (32: 65%, 33: 92%) |
| Client 2 | **85.0%** (4: 94%, 5: 76%) | **77.0%** (14: 58%, 15: 96%) | **84.5%** (24: 82%, 25: 87%) | **67.5%** (34: 99%, 35: 36%) |
| Client 3 | **80.0%** (6: 77%, 7: 83%) | **89.5%** (16: 83%, 17: 96%) | **75.5%** (26: 52%, 27: 99%) | **84.0%** (36: 78%, 37: 90%) |
| Client 4 | **82.0%** (8: 85%, 9: 79%) | **79.5%** (18: 85%, 19: 74%) | **78.0%** (28: 71%, 29: 85%) | **80.5%** (38: 73%, 39: 88%) |
