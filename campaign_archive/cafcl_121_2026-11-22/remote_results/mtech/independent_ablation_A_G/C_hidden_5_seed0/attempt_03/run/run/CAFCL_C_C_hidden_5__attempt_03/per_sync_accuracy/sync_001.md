# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.50% on 10 observed labels.
Validation accuracy: 0.518; seen labels never predicted: 3.
Server-accepted classes: 8; pending private classes: 2.
Accepted-only TEST accuracy: 0.51875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (8: 100%, 9: 0%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **52.0%** (0: 4%, 1: 100%) |
| Client 4 | **55.5%** (2: 13%, 3: 98%) |
