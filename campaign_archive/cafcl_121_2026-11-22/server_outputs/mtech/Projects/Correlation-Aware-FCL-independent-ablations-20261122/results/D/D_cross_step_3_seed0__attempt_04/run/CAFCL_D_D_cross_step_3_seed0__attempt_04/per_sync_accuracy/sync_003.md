# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.83% on 18 observed labels.
Validation accuracy: 0.7444444444444445; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.7383333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 79%, 1: 87%) | **60.5%** (10: 77%, 11: 44%) | **69.5%** (14: 41%, 15: 98%) |
| Client 1 | **90.0%** (2: 88%, 3: 92%) | **82.5%** (4: 79%, 5: 86%) | **90.0%** (2: 88%, 3: 92%) |
| Client 2 | **82.5%** (4: 79%, 5: 86%) | **68.5%** (8: 97%, 9: 40%) | **59.5%** (12: 97%, 13: 22%) |
| Client 3 | **69.0%** (6: 46%, 7: 92%) | **59.5%** (12: 97%, 13: 22%) | **82.0%** (16: 73%, 17: 91%) |
| Client 4 | **68.5%** (8: 97%, 9: 40%) | **90.0%** (2: 88%, 3: 92%) | **60.5%** (10: 77%, 11: 44%) |
