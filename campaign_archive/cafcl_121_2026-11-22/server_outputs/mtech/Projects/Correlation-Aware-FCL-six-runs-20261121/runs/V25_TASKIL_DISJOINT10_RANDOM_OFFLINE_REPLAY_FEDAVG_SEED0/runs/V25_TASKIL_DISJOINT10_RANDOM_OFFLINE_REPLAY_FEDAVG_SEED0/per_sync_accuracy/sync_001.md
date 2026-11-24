# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.10% on 10 observed labels.
Validation accuracy: 0.5; seen labels never predicted: 4.
Server-accepted classes: 8; pending private classes: 2.
Accepted-only TEST accuracy: 0.50125.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **50.5%** (2: 1%, 3: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) |
| Client 4 | **50.0%** (8: 0%, 9: 100%) [PENDING upload] |
