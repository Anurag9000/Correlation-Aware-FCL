# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.10% on 10 observed labels.
Validation accuracy: 0.142; seen labels never predicted: 6.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.141.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **47.5%** (0: 44%, 1: 51%) |
| Client 1 | **21.5%** (2: 43%, 3: 0%) |
| Client 2 | **1.5%** (4: 3%, 5: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) |
| Client 4 | **0.0%** (8: 0%, 9: 0%) |
