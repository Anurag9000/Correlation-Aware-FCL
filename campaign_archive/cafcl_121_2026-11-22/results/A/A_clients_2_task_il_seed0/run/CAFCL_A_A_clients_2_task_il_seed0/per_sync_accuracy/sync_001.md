# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.25% on 4 observed labels.
Validation accuracy: 0.51; seen labels never predicted: 1.
Server-accepted classes: 4; pending private classes: 0.
Accepted-only TEST accuracy: 0.5025.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.5%** (0: 1%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
