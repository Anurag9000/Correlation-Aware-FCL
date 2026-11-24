# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.92% on 24 observed labels.
Validation accuracy: 0.8075; seen labels never predicted: 0.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.7891666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **84.5%** (2: 73%, 3: 96%) | **79.5%** (8: 93%, 9: 66%) | **57.5%** (18: 18%, 19: 97%) |
| Client 1 | **87.5%** (0: 90%, 1: 85%) | **73.5%** (10: 83%, 11: 64%) | **89.5%** (16: 80%, 17: 99%) |
| Client 2 | **77.5%** (6: 74%, 7: 81%) | **79.5%** (8: 93%, 9: 66%) | **82.0%** (22: 77%, 23: 87%) |
| Client 3 | **87.5%** (0: 90%, 1: 85%) | **72.5%** (14: 46%, 15: 99%) | **90.5%** (20: 81%, 21: 100%) |
| Client 4 | **85.0%** (4: 93%, 5: 77%) | **67.5%** (12: 95%, 13: 40%) | **89.5%** (16: 80%, 17: 99%) |
