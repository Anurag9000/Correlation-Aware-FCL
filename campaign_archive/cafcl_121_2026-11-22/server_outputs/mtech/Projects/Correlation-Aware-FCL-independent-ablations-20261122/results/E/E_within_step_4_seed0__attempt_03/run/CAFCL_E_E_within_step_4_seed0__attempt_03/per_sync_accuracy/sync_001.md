# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 62.00% on 4 observed labels.
Validation accuracy: 0.6; seen labels never predicted: 0.
Server-accepted classes: 4; pending private classes: 0.
Accepted-only TEST accuracy: 0.62.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **61.0%** (0: 25%, 1: 97%) |
| Client 1 | **61.0%** (0: 25%, 1: 97%) |
| Client 2 | **63.0%** (2: 26%, 3: 100%) |
| Client 3 | **61.0%** (0: 25%, 1: 97%) |
| Client 4 | **61.0%** (0: 25%, 1: 97%) |
