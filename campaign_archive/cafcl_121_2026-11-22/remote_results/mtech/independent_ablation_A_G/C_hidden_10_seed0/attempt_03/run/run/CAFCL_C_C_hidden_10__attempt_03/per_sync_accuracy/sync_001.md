# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.50% on 10 observed labels.
Validation accuracy: 0.518; seen labels never predicted: 3.
Server-accepted classes: 8; pending private classes: 2.
Accepted-only TEST accuracy: 0.53125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (8: 100%, 9: 0%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] |
| Client 2 | **58.5%** (4: 47%, 5: 70%) |
| Client 3 | **54.0%** (0: 8%, 1: 100%) |
| Client 4 | **50.0%** (2: 0%, 3: 100%) |
