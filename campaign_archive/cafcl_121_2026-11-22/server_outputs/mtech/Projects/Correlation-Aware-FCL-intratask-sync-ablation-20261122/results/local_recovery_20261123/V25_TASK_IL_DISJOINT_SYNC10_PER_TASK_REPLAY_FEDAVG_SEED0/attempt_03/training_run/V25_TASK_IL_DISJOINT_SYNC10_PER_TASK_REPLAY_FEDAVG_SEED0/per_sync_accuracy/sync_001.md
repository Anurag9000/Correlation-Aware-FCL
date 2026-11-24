# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.60% on 10 observed labels.
Validation accuracy: 0.548; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.556.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **76.0%** (0: 93%, 1: 59%) |
| Client 1 | **51.0%** (2: 96%, 3: 6%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **50.0%** (6: 100%, 7: 0%) |
| Client 4 | **51.0%** (8: 99%, 9: 3%) |
