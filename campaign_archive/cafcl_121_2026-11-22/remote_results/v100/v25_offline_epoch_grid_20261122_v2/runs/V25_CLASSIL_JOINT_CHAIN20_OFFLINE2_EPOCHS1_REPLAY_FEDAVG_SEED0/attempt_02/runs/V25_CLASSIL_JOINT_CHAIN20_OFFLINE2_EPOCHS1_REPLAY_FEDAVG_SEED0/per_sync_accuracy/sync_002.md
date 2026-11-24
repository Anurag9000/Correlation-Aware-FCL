# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.27% on 11 observed labels.
Validation accuracy: 0.19818181818181818; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.20272727272727273.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **54.0%** (0: 67%, 1: 41%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **36.0%** (1: 41%, 2: 31%) | **12.0%** (6: 0%, 7: 24%) |
| Client 2 | **15.5%** (2: 31%, 3: 0%) | **12.0%** (7: 24%, 8: 0%) |
| Client 3 | **16.5%** (3: 0%, 4: 33%) | **2.0%** (8: 0%, 9: 4%) |
| Client 4 | **16.5%** (4: 33%, 5: 0%) | **13.5%** (9: 4%, 10: 23%) |
