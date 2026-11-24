# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.40% on 30 observed labels.
Validation accuracy: 0.7453333333333333; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.754.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **77.0%** (0: 54%, 1: 100%) | **55.5%** (10: 96%, 11: 15%) | **87.0%** (20: 74%, 21: 100%) |
| Client 1 | **79.5%** (2: 59%, 3: 100%) | **61.0%** (12: 97%, 13: 25%) | **85.0%** (22: 79%, 23: 91%) |
| Client 2 | **82.5%** (4: 89%, 5: 76%) | **76.0%** (14: 58%, 15: 94%) | **82.0%** (24: 80%, 25: 84%) |
| Client 3 | **63.5%** (6: 33%, 7: 94%) | **87.5%** (16: 78%, 17: 97%) | **70.0%** (26: 42%, 27: 98%) |
| Client 4 | **72.5%** (8: 97%, 9: 48%) | **75.0%** (18: 83%, 19: 67%) | **77.0%** (28: 68%, 29: 86%) |
