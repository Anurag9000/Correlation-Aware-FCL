# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.90% on 10 observed labels.
Validation accuracy: 0.122; seen labels never predicted: 6.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.129.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **40.0%** (0: 28%, 1: 52%) |
| Client 1 | **24.0%** (2: 48%, 3: 0%) |
| Client 2 | **0.5%** (4: 1%, 5: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) |
| Client 4 | **0.0%** (8: 0%, 9: 0%) |
