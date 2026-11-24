# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 57.65% on 20 observed labels.
Validation accuracy: 0.572; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5765.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.0%** (0: 16%, 1: 100%) | **50.5%** (10: 99%, 11: 2%) |
| Client 1 | **50.5%** (2: 1%, 3: 100%) | **64.5%** (12: 42%, 13: 87%) |
| Client 2 | **67.0%** (4: 97%, 5: 37%) | **50.0%** (14: 0%, 15: 100%) |
| Client 3 | **56.0%** (6: 15%, 7: 97%) | **68.5%** (16: 40%, 17: 97%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **60.0%** (18: 88%, 19: 32%) |
