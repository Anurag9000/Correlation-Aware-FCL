# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.88% on 16 observed labels.
Validation accuracy: 0.62125; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.60875.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **57.0%** (2: 14%, 3: 100%) | **53.0%** (8: 99%, 9: 7%) |
| Client 1 | **78.0%** (0: 64%, 1: 92%) | **51.0%** (10: 99%, 11: 3%) |
| Client 2 | **56.0%** (6: 14%, 7: 98%) | **53.0%** (8: 99%, 9: 7%) |
| Client 3 | **78.0%** (0: 64%, 1: 92%) | **63.5%** (14: 82%, 15: 45%) |
| Client 4 | **73.0%** (4: 98%, 5: 48%) | **55.5%** (12: 16%, 13: 95%) |
