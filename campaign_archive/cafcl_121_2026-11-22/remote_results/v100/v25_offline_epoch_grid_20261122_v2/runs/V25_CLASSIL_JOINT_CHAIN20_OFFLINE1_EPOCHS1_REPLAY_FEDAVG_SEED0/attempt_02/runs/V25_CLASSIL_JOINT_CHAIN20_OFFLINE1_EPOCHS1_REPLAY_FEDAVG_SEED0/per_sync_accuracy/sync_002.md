# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.55% on 11 observed labels.
Validation accuracy: 0.1890909090909091; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.21545454545454545.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **39.5%** (0: 71%, 1: 8%) | **6.5%** (5: 3%, 6: 10%) |
| Client 1 | **29.5%** (1: 8%, 2: 51%) | **5.5%** (6: 10%, 7: 1%) |
| Client 2 | **34.0%** (2: 51%, 3: 17%) | **0.5%** (7: 1%, 8: 0%) |
| Client 3 | **45.0%** (3: 17%, 4: 73%) | **0.5%** (8: 0%, 9: 1%) |
| Client 4 | **38.0%** (4: 73%, 5: 3%) | **1.5%** (9: 1%, 10: 2%) |
