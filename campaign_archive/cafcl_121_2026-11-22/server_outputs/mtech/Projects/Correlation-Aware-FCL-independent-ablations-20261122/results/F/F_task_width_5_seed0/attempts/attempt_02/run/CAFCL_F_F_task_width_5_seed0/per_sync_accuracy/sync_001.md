# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.32% on 25 observed labels.
Validation accuracy: 0.2144; seen labels never predicted: 15.
Server-accepted classes: 25; pending private classes: 0.
Accepted-only TEST accuracy: 0.2132.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **20.0%** (0: 0%, 1: 0%, 2: 100%, 3: 0%, 4: 0%) |
| Client 1 | **20.8%** (5: 100%, 6: 0%, 7: 0%, 8: 4%, 9: 0%) |
| Client 2 | **22.0%** (10: 14%, 11: 0%, 12: 96%, 13: 0%, 14: 0%) |
| Client 3 | **23.2%** (15: 22%, 16: 0%, 17: 0%, 18: 0%, 19: 94%) |
| Client 4 | **20.6%** (20: 0%, 21: 0%, 22: 100%, 23: 3%, 24: 0%) |
