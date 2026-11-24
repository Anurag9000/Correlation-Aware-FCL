# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.05% on 20 observed labels.
Validation accuracy: 0.262; seen labels never predicted: 10.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.263125.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **25.2%** (0: 0%, 1: 0%, 2: 1%, 3: 100%) |
| Client 1 | **28.0%** (4: 97%, 5: 0%, 6: 0%, 7: 15%) |
| Client 2 | **25.5%** (8: 100%, 9: 0%, 10: 0%, 11: 2%) |
| Client 3 | **26.5%** (12: 5%, 13: 0%, 14: 1%, 15: 100%) |
| Client 4 | **25.0%** (16: 0%, 17: 100%, 18: 0%, 19: 0%) [PENDING upload] |
