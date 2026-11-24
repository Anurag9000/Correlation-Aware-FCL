# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.45% on 11 observed labels.
Validation accuracy: 0.18181818181818182; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.21454545454545454.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **47.0%** (0: 75%, 1: 19%) | **3.5%** (5: 0%, 6: 7%) |
| Client 1 | **13.5%** (1: 19%, 2: 8%) | **3.5%** (6: 7%, 7: 0%) |
| Client 2 | **25.0%** (2: 8%, 3: 42%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **58.5%** (3: 42%, 4: 75%) | **1.5%** (8: 0%, 9: 3%) |
| Client 4 | **37.5%** (4: 75%, 5: 0%) | **5.0%** (9: 3%, 10: 7%) |
