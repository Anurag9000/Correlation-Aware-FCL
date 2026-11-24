# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.62% on 8 observed labels.
Validation accuracy: 0.55; seen labels never predicted: 2.
Server-accepted classes: 8; pending private classes: 0.
Accepted-only TEST accuracy: 0.54625.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **52.0%** (2: 4%, 3: 100%) |
| Client 1 | **66.5%** (0: 38%, 1: 95%) |
| Client 2 | **50.0%** (6: 0%, 7: 100%) |
| Client 3 | **66.5%** (0: 38%, 1: 95%) |
| Client 4 | **50.0%** (4: 100%, 5: 0%) |
