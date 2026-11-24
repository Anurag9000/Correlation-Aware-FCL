# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 57.63% on 8 observed labels.
Validation accuracy: 0.545; seen labels never predicted: 0.
Server-accepted classes: 8; pending private classes: 0.
Accepted-only TEST accuracy: 0.57625.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **74.5%** (0: 93%, 1: 56%) | **50.5%** (4: 100%, 5: 1%) |
| Client 1 | **52.0%** (2: 4%, 3: 100%) | **53.5%** (6: 9%, 7: 98%) |
