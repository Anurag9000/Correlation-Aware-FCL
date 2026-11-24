# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.64% on 11 observed labels.
Validation accuracy: 0.21818181818181817; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.24636363636363637.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **49.0%** (0: 58%, 1: 40%) | **8.5%** (5: 1%, 6: 16%) |
| Client 1 | **36.0%** (1: 40%, 2: 32%) | **8.0%** (6: 16%, 7: 0%) |
| Client 2 | **27.5%** (2: 32%, 3: 23%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **50.5%** (3: 23%, 4: 78%) | **11.5%** (8: 0%, 9: 23%) |
| Client 4 | **39.5%** (4: 78%, 5: 1%) | **11.5%** (9: 23%, 10: 0%) |
