# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.61% on 18 observed labels.
Validation accuracy: 0.7666666666666667; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.7561111111111111.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **80.0%** (0: 70%, 1: 90%) | **76.5%** (6: 86%, 7: 67%) | **59.5%** (12: 99%, 13: 20%) |
| Client 1 | **85.5%** (2: 73%, 3: 98%) | **71.0%** (8: 95%, 9: 47%) | **68.0%** (14: 40%, 15: 96%) |
| Client 2 | **82.5%** (4: 89%, 5: 76%) | **67.5%** (10: 90%, 11: 45%) | **90.0%** (16: 84%, 17: 96%) |
