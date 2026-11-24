# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.60% on 10 observed labels.
Validation accuracy: 0.556; seen labels never predicted: 3.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.536.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **50.0%** (4: 0%, 5: 100%) |
| Client 3 | **66.5%** (6: 39%, 7: 94%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) |
