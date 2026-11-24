# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 49.17% on 6 observed labels.
Validation accuracy: 0.52; seen labels never predicted: 0.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.49166666666666664.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **84.0%** (0: 88%, 1: 80%) |
| Client 1 | **41.0%** (1: 80%, 2: 2%) |
| Client 2 | **2.0%** (2: 2%, 3: 2%) |
| Client 3 | **47.0%** (3: 2%, 4: 92%) |
| Client 4 | **61.5%** (4: 92%, 5: 31%) |
