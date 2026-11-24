# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.36% on 25 observed labels.
Validation accuracy: 0.22; seen labels never predicted: 15.
Server-accepted classes: 25; pending private classes: 0.
Accepted-only TEST accuracy: 0.2136.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **20.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%, 4: 100%) |
| Client 1 | **20.4%** (5: 0%, 6: 2%, 7: 100%, 8: 0%, 9: 0%) |
| Client 2 | **20.0%** (10: 2%, 11: 0%, 12: 98%, 13: 0%, 14: 0%) |
| Client 3 | **20.0%** (15: 100%, 16: 0%, 17: 0%, 18: 0%, 19: 0%) |
| Client 4 | **26.4%** (20: 0%, 21: 0%, 22: 92%, 23: 40%, 24: 0%) |
