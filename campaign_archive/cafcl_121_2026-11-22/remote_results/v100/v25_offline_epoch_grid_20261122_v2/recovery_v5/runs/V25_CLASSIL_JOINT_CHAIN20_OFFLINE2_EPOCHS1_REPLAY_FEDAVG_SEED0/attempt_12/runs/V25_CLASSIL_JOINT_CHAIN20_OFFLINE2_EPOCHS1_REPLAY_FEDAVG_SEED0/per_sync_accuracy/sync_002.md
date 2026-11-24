# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.73% on 11 observed labels.
Validation accuracy: 0.24181818181818182; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.22727272727272727.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.0%** (0: 58%, 1: 58%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **42.0%** (1: 58%, 2: 26%) | **8.5%** (6: 0%, 7: 17%) |
| Client 2 | **13.0%** (2: 26%, 3: 0%) | **8.5%** (7: 17%, 8: 0%) |
| Client 3 | **33.0%** (3: 0%, 4: 66%) | **8.5%** (8: 0%, 9: 17%) |
| Client 4 | **33.0%** (4: 66%, 5: 0%) | **12.5%** (9: 17%, 10: 8%) |
