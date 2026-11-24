# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.25% on 20 observed labels.
Validation accuracy: 0.553; seen labels never predicted: 2.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5525.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **63.5%** (0: 27%, 1: 100%) | **50.5%** (10: 99%, 11: 2%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **56.0%** (12: 22%, 13: 90%) |
| Client 2 | **68.0%** (4: 97%, 5: 39%) | **50.5%** (14: 1%, 15: 100%) |
| Client 3 | **52.0%** (6: 8%, 7: 96%) | **61.0%** (16: 23%, 17: 99%) |
| Client 4 | **51.0%** (8: 99%, 9: 3%) | **50.0%** (18: 100%, 19: 0%) |
