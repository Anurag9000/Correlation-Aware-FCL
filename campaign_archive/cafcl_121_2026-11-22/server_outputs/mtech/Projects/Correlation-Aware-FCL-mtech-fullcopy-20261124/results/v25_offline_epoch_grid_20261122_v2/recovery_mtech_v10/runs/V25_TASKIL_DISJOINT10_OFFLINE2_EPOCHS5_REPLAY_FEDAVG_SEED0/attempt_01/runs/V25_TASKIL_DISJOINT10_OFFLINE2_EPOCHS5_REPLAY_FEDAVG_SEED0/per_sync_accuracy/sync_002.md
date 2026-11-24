# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.45% on 20 observed labels.
Validation accuracy: 0.518; seen labels never predicted: 2.
Server-accepted classes: 14; pending private classes: 6.
Accepted-only TEST accuracy: 0.52.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **60.5%** (0: 21%, 1: 100%) | **51.0%** (10: 97%, 11: 5%) [PENDING upload] |
| Client 1 | **52.0%** (2: 4%, 3: 100%) | **49.5%** (12: 99%, 13: 0%) |
| Client 2 | **49.0%** (4: 0%, 5: 98%) [PENDING upload] | **50.5%** (14: 1%, 15: 100%) [PENDING upload] |
| Client 3 | **51.0%** (6: 3%, 7: 99%) | **50.0%** (16: 100%, 17: 0%) |
| Client 4 | **51.0%** (8: 99%, 9: 3%) | **50.0%** (18: 100%, 19: 0%) |
