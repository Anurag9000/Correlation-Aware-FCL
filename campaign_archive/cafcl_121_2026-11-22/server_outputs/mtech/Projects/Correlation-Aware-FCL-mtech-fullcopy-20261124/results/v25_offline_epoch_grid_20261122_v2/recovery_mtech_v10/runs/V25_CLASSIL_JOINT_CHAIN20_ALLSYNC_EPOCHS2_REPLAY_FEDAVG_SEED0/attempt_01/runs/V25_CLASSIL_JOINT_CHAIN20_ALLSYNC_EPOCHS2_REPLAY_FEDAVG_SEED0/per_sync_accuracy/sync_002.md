# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.00% on 11 observed labels.
Validation accuracy: 0.20363636363636364; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.21.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **18.0%** (0: 34%, 1: 2%) | **16.0%** (5: 26%, 6: 6%) |
| Client 1 | **18.0%** (1: 2%, 2: 34%) | **5.5%** (6: 6%, 7: 5%) |
| Client 2 | **26.5%** (2: 34%, 3: 19%) | **16.0%** (7: 5%, 8: 27%) |
| Client 3 | **47.0%** (3: 19%, 4: 75%) | **15.0%** (8: 27%, 9: 3%) |
| Client 4 | **50.5%** (4: 75%, 5: 26%) | **1.5%** (9: 3%, 10: 0%) |
