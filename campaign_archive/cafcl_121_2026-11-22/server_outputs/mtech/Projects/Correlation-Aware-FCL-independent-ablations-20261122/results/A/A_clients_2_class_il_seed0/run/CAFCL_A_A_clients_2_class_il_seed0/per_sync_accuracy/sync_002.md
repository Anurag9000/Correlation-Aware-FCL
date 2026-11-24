# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.38% on 8 observed labels.
Validation accuracy: 0.255; seen labels never predicted: 0.
Server-accepted classes: 8; pending private classes: 0.
Accepted-only TEST accuracy: 0.27375.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **24.0%** (0: 39%, 1: 9%) | **0.5%** (4: 0%, 5: 1%) |
| Client 1 | **84.5%** (2: 70%, 3: 99%) | **0.5%** (6: 1%, 7: 0%) |
