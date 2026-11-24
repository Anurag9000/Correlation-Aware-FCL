# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.15% on 20 observed labels.
Validation accuracy: 0.589; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5915.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **63.5%** (0: 27%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **51.5%** (2: 3%, 3: 100%) | **54.0%** (12: 89%, 13: 19%) |
| Client 2 | **73.5%** (4: 80%, 5: 67%) | **59.5%** (14: 19%, 15: 100%) |
| Client 3 | **64.0%** (6: 38%, 7: 90%) | **73.5%** (16: 54%, 17: 93%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **50.5%** (18: 99%, 19: 2%) |
