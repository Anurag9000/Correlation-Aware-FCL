# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.36% on 11 observed labels.
Validation accuracy: 0.18; seen labels never predicted: 3.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.19363636363636363.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **21.0%** (0: 39%, 1: 3%) | **5.0%** (5: 0%, 6: 10%) |
| Client 1 | **33.5%** (1: 3%, 2: 64%) | **5.0%** (6: 10%, 7: 0%) |
| Client 2 | **39.5%** (2: 64%, 3: 15%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **46.5%** (3: 15%, 4: 78%) | **2.0%** (8: 0%, 9: 4%) |
| Client 4 | **39.0%** (4: 78%, 5: 0%) | **2.0%** (9: 4%, 10: 0%) |
