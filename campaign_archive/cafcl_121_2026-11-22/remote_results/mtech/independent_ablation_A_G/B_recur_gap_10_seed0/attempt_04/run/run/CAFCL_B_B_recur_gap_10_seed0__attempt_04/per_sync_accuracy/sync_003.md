# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.00% on 30 observed labels.
Validation accuracy: 0.7546666666666667; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.75.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 56%, 1: 97%) | **61.0%** (10: 97%, 11: 25%) | **89.5%** (20: 81%, 21: 98%) |
| Client 1 | **69.0%** (2: 39%, 3: 99%) | **59.5%** (12: 88%, 13: 31%) | **86.5%** (22: 80%, 23: 93%) |
| Client 2 | **81.5%** (4: 99%, 5: 64%) | **70.0%** (14: 45%, 15: 95%) | **83.0%** (24: 85%, 25: 81%) |
| Client 3 | **61.0%** (6: 28%, 7: 94%) | **89.0%** (16: 81%, 17: 97%) | **69.0%** (26: 39%, 27: 99%) |
| Client 4 | **76.5%** (8: 93%, 9: 60%) | **76.5%** (18: 80%, 19: 73%) | **76.5%** (28: 63%, 29: 90%) |
