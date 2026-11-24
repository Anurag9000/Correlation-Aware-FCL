# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.07% on 28 observed labels.
Validation accuracy: 0.8335714285714285; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.8207142857142857.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **82.0%** (0: 76%, 1: 88%) | **78.5%** (8: 95%, 9: 62%) | **92.0%** (16: 89%, 17: 95%) | **83.0%** (22: 72%, 23: 94%) |
| Client 1 | **86.5%** (2: 80%, 3: 93%) | **68.0%** (10: 78%, 11: 58%) | **78.5%** (8: 95%, 9: 62%) | **92.0%** (16: 89%, 17: 95%) |
| Client 2 | **85.0%** (4: 82%, 5: 88%) | **69.0%** (12: 95%, 13: 43%) | **82.0%** (18: 82%, 19: 82%) | **94.0%** (20: 89%, 21: 99%) |
| Client 3 | **80.5%** (6: 69%, 7: 92%) | **82.0%** (14: 72%, 15: 92%) | **94.0%** (20: 89%, 21: 99%) | **89.0%** (24: 85%, 25: 93%) |
| Client 4 | **78.5%** (8: 95%, 9: 62%) | **85.0%** (4: 82%, 5: 88%) | **69.0%** (12: 95%, 13: 43%) | **77.5%** (26: 59%, 27: 96%) |
