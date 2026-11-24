# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.83% on 30 observed labels.
Validation accuracy: 0.8533333333333334; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.8383333333333334.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 85%, 1: 83%) | **83.0%** (6: 90%, 7: 76%) | **67.5%** (12: 97%, 13: 38%) | **79.0%** (18: 76%, 19: 82%) | **85.5%** (24: 89%, 25: 82%) |
| Client 1 | **91.0%** (2: 87%, 3: 95%) | **88.0%** (8: 90%, 9: 86%) | **85.0%** (14: 74%, 15: 96%) | **95.0%** (20: 91%, 21: 99%) | **77.0%** (26: 60%, 27: 94%) |
| Client 2 | **88.5%** (4: 91%, 5: 86%) | **77.0%** (10: 85%, 11: 69%) | **93.0%** (16: 90%, 17: 96%) | **86.5%** (22: 75%, 23: 98%) | **77.5%** (28: 82%, 29: 73%) |
