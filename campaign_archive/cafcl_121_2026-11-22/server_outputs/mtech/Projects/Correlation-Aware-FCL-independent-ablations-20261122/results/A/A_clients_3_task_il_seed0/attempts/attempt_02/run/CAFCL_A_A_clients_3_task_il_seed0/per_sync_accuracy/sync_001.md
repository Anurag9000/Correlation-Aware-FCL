# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.17% on 6 observed labels.
Validation accuracy: 0.5066666666666667; seen labels never predicted: 2.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.5016666666666667.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.5%** (0: 1%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
