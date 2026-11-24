# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.70% on 20 observed labels.
Validation accuracy: 0.621; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.607.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.5%** (0: 3%, 1: 100%) | **50.0%** (10: 99%, 11: 1%) |
| Client 1 | **56.0%** (2: 12%, 3: 100%) | **64.0%** (12: 76%, 13: 52%) |
| Client 2 | **71.5%** (4: 92%, 5: 51%) | **69.5%** (14: 75%, 15: 64%) |
| Client 3 | **61.0%** (6: 37%, 7: 85%) | **79.5%** (16: 64%, 17: 95%) |
| Client 4 | **51.0%** (8: 100%, 9: 2%) | **53.0%** (18: 18%, 19: 88%) |
