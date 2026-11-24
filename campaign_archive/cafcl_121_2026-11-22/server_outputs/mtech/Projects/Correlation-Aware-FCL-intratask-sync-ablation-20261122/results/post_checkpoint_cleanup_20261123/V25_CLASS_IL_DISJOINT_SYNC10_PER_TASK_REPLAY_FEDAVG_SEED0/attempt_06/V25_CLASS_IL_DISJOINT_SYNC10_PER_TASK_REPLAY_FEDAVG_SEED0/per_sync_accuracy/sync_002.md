# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.00% on 10 observed labels.
Validation accuracy: 0.126; seen labels never predicted: 4.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.13.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **7.0%** (0: 14%, 1: 0%) | **7.0%** (0: 14%, 1: 0%) |
| Client 1 | **47.5%** (2: 0%, 3: 95%) | **47.5%** (2: 0%, 3: 95%) |
| Client 2 | **0.5%** (4: 0%, 5: 1%) | **0.5%** (4: 0%, 5: 1%) |
| Client 3 | **0.5%** (6: 1%, 7: 0%) | **0.5%** (6: 1%, 7: 0%) |
| Client 4 | **9.5%** (8: 18%, 9: 1%) | **9.5%** (8: 18%, 9: 1%) |
