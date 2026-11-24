# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 49.82% on 11 observed labels.
Validation accuracy: 0.49454545454545457; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5045454545454545.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **69.5%** (0: 78%, 1: 61%) | **54.0%** (5: 77%, 6: 31%) |
| Client 1 | **58.5%** (1: 61%, 2: 56%) | **51.0%** (6: 31%, 7: 71%) |
| Client 2 | **29.0%** (2: 56%, 3: 2%) | **70.5%** (7: 71%, 8: 70%) |
| Client 3 | **32.5%** (3: 2%, 4: 63%) | **54.5%** (8: 70%, 9: 39%) |
| Client 4 | **70.0%** (4: 63%, 5: 77%) | **19.5%** (9: 39%, 10: 0%) |
