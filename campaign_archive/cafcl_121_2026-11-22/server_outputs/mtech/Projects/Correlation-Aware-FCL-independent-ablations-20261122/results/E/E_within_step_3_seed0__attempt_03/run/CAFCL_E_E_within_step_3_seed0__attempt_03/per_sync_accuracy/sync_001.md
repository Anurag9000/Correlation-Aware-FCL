# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.50% on 6 observed labels.
Validation accuracy: 0.51; seen labels never predicted: 2.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.505.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **51.5%** (0: 3%, 1: 100%) |
| Client 1 | **51.5%** (0: 3%, 1: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **51.5%** (0: 3%, 1: 100%) |
| Client 4 | **50.0%** (2: 0%, 3: 100%) |
