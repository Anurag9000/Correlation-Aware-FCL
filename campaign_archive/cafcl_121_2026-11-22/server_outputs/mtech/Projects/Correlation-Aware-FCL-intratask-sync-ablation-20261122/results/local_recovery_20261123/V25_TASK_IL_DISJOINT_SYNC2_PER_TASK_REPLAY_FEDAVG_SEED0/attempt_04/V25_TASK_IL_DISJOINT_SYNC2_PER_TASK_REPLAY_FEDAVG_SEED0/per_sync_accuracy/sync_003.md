# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.60% on 20 observed labels.
Validation accuracy: 0.728; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.726.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 78%, 1: 88%) | **83.0%** (0: 78%, 1: 88%) | **62.5%** (10: 78%, 11: 47%) |
| Client 1 | **87.0%** (2: 75%, 3: 99%) | **87.0%** (2: 75%, 3: 99%) | **59.5%** (12: 90%, 13: 29%) |
| Client 2 | **80.5%** (4: 89%, 5: 72%) | **80.5%** (4: 89%, 5: 72%) | **71.0%** (14: 73%, 15: 69%) |
| Client 3 | **71.5%** (6: 63%, 7: 80%) | **71.5%** (6: 63%, 7: 80%) | **79.5%** (16: 83%, 17: 76%) |
| Client 4 | **62.0%** (8: 96%, 9: 28%) | **62.0%** (8: 96%, 9: 28%) | **69.5%** (18: 85%, 19: 54%) |
