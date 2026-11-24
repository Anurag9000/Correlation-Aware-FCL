# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.20% on 10 observed labels.
Validation accuracy: 0.612; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.602.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **71.0%** (0: 44%, 1: 98%) | **71.0%** (0: 44%, 1: 98%) |
| Client 1 | **52.0%** (2: 4%, 3: 100%) | **52.0%** (2: 4%, 3: 100%) |
| Client 2 | **60.0%** (4: 98%, 5: 22%) | **60.0%** (4: 98%, 5: 22%) |
| Client 3 | **66.5%** (6: 45%, 7: 88%) | **66.5%** (6: 45%, 7: 88%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **51.5%** (8: 99%, 9: 4%) |
