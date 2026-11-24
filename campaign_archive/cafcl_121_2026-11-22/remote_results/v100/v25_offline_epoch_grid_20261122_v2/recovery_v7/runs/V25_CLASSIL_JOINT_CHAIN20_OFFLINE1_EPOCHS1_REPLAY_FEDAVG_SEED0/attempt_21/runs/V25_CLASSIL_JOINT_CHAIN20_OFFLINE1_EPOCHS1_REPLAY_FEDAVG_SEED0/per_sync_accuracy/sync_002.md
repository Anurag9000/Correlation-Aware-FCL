# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.55% on 11 observed labels.
Validation accuracy: 0.21272727272727274; seen labels never predicted: 3.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.24545454545454545.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (0: 72%, 1: 28%) | **0.5%** (5: 0%, 6: 1%) |
| Client 1 | **23.5%** (1: 28%, 2: 19%) | **3.0%** (6: 1%, 7: 5%) |
| Client 2 | **32.0%** (2: 19%, 3: 45%) | **2.5%** (7: 5%, 8: 0%) |
| Client 3 | **59.5%** (3: 45%, 4: 74%) | **13.0%** (8: 0%, 9: 26%) |
| Client 4 | **37.0%** (4: 74%, 5: 0%) | **13.0%** (9: 26%, 10: 0%) |
