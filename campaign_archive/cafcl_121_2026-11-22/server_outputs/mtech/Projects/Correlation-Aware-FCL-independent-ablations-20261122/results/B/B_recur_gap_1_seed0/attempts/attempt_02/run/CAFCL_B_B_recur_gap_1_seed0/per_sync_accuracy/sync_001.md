# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.50% on 10 observed labels.
Validation accuracy: 0.528; seen labels never predicted: 4.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.505.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **52.5%** (6: 64%, 7: 41%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) |
