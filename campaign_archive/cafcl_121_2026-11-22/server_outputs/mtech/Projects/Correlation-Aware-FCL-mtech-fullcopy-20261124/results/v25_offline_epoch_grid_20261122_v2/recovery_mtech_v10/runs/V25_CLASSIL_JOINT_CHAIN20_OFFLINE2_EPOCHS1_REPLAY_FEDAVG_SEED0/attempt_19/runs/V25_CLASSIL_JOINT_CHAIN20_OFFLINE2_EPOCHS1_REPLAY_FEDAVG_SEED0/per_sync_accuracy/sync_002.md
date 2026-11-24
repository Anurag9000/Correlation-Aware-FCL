# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.27% on 11 observed labels.
Validation accuracy: 0.18363636363636363; seen labels never predicted: 5.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.20272727272727273.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **55.0%** (0: 57%, 1: 53%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **36.5%** (1: 53%, 2: 20%) | **8.5%** (6: 0%, 7: 17%) |
| Client 2 | **10.0%** (2: 20%, 3: 0%) | **8.5%** (7: 17%, 8: 0%) |
| Client 3 | **23.5%** (3: 0%, 4: 47%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **23.5%** (4: 47%, 5: 0%) | **14.5%** (9: 0%, 10: 29%) |
