# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.27% on 11 observed labels.
Validation accuracy: 0.22181818181818183; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.23272727272727273.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **52.0%** (0: 77%, 1: 27%) | **0.5%** (5: 0%, 6: 1%) |
| Client 1 | **36.5%** (1: 27%, 2: 46%) | **1.5%** (6: 1%, 7: 2%) |
| Client 2 | **37.5%** (2: 46%, 3: 29%) | **1.0%** (7: 2%, 8: 0%) |
| Client 3 | **50.5%** (3: 29%, 4: 72%) | **1.0%** (8: 0%, 9: 2%) |
| Client 4 | **36.0%** (4: 72%, 5: 0%) | **1.0%** (9: 2%, 10: 0%) |
