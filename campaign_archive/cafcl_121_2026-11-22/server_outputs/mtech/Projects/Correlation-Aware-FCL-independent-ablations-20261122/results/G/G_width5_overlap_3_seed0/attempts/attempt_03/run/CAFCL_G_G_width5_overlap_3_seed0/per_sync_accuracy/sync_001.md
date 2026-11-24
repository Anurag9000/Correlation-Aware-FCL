# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.15% on 13 observed labels.
Validation accuracy: 0.23076923076923078; seen labels never predicted: 7.
Server-accepted classes: 13; pending private classes: 0.
Accepted-only TEST accuracy: 0.23153846153846153.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **20.0%** (4: 0%, 7: 100%, 15: 0%, 58: 0%, 63: 0%) |
| Client 1 | **20.0%** (4: 0%, 40: 0%, 69: 0%, 80: 100%, 95: 0%) |
| Client 2 | **0.0%** (4: 0%, 15: 0%, 40: 0%, 58: 0%, 95: 0%) |
| Client 3 | **0.0%** (4: 0%, 16: 0%, 40: 0%, 86: 0%, 95: 0%) |
| Client 4 | **20.2%** (4: 0%, 12: 1%, 16: 0%, 18: 100%, 86: 0%) |
