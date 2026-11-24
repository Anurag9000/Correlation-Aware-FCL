# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.90% on 10 observed labels.
Validation accuracy: 0.552; seen labels never predicted: 1.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.549.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **68.5%** (0: 99%, 1: 38%) |
| Client 1 | **52.0%** (2: 99%, 3: 5%) |
| Client 2 | **52.5%** (4: 100%, 5: 5%) |
| Client 3 | **50.0%** (6: 100%, 7: 0%) |
| Client 4 | **51.5%** (8: 100%, 9: 3%) |
