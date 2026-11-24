# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.82% on 11 observed labels.
Validation accuracy: 0.18545454545454546; seen labels never predicted: 5.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.19818181818181818.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **70.0%** (0: 83%, 1: 57%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **34.0%** (1: 57%, 2: 11%) | **5.0%** (6: 0%, 7: 10%) |
| Client 2 | **5.5%** (2: 11%, 3: 0%) | **5.0%** (7: 10%, 8: 0%) |
| Client 3 | **21.5%** (3: 0%, 4: 43%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **21.5%** (4: 43%, 5: 0%) | **7.0%** (9: 0%, 10: 14%) |
