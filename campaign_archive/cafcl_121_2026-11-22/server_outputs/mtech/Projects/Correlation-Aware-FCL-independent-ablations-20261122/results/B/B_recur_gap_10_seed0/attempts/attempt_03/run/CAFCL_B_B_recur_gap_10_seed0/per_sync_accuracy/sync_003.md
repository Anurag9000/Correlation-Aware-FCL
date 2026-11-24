# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.20% on 30 observed labels.
Validation accuracy: 0.7726666666666666; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.772.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 66%, 1: 96%) | **64.5%** (10: 93%, 11: 36%) | **91.5%** (20: 83%, 21: 100%) |
| Client 1 | **77.5%** (2: 56%, 3: 99%) | **62.5%** (12: 97%, 13: 28%) | **87.0%** (22: 79%, 23: 95%) |
| Client 2 | **84.5%** (4: 96%, 5: 73%) | **73.5%** (14: 50%, 15: 97%) | **83.0%** (24: 86%, 25: 80%) |
| Client 3 | **73.0%** (6: 54%, 7: 92%) | **85.5%** (16: 73%, 17: 98%) | **68.0%** (26: 37%, 27: 99%) |
| Client 4 | **76.0%** (8: 98%, 9: 54%) | **78.0%** (18: 80%, 19: 76%) | **72.5%** (28: 52%, 29: 93%) |
