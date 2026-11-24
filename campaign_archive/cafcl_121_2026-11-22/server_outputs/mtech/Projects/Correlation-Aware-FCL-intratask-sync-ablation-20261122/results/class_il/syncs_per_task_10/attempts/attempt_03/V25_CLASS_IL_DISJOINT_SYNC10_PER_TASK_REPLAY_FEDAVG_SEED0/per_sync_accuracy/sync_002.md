# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.30% on 10 observed labels.
Validation accuracy: 0.16; seen labels never predicted: 3.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.153.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **12.5%** (0: 25%, 1: 0%) | **12.5%** (0: 25%, 1: 0%) |
| Client 1 | **42.5%** (2: 0%, 3: 85%) | **42.5%** (2: 0%, 3: 85%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (4: 0%, 5: 0%) |
| Client 3 | **0.5%** (6: 0%, 7: 1%) | **0.5%** (6: 0%, 7: 1%) |
| Client 4 | **21.0%** (8: 42%, 9: 0%) | **21.0%** (8: 42%, 9: 0%) |
