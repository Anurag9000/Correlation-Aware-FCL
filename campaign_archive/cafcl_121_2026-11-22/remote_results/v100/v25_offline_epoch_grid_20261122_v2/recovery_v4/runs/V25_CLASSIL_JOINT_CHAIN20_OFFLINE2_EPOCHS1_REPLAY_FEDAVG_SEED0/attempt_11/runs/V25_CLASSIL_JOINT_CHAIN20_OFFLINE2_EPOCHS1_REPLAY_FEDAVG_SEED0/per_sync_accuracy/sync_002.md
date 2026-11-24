# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.27% on 11 observed labels.
Validation accuracy: 0.2018181818181818; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.22272727272727272.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **53.0%** (0: 51%, 1: 55%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **48.0%** (1: 55%, 2: 41%) | **11.0%** (6: 0%, 7: 22%) |
| Client 2 | **20.5%** (2: 41%, 3: 0%) | **11.0%** (7: 22%, 8: 0%) |
| Client 3 | **21.0%** (3: 0%, 4: 42%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **21.0%** (4: 42%, 5: 0%) | **17.0%** (9: 0%, 10: 34%) |
