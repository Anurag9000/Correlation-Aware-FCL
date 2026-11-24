# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.00% on 10 observed labels.
Validation accuracy: 0.1; seen labels never predicted: 9.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.1.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) |
| Client 4 | **0.0%** (8: 0%, 9: 0%) |
