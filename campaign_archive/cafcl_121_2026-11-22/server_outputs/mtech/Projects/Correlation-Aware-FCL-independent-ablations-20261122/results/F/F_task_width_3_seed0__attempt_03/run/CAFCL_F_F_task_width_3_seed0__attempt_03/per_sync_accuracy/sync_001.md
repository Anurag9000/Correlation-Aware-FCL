# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.40% on 15 observed labels.
Validation accuracy: 0.3506666666666667; seen labels never predicted: 7.
Server-accepted classes: 15; pending private classes: 0.
Accepted-only TEST accuracy: 0.344.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **33.3%** (0: 0%, 1: 0%, 2: 100%) |
| Client 1 | **33.3%** (3: 100%, 4: 0%, 5: 0%) |
| Client 2 | **36.0%** (6: 0%, 7: 12%, 8: 96%) |
| Client 3 | **33.7%** (9: 0%, 10: 82%, 11: 19%) |
| Client 4 | **35.7%** (12: 27%, 13: 80%, 14: 0%) |
