# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.82% on 11 observed labels.
Validation accuracy: 0.11636363636363636; seen labels never predicted: 6.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.12818181818181817.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.5%** (0: 1%, 1: 0%) | **3.5%** (5: 0%, 6: 7%) |
| Client 1 | **0.5%** (1: 0%, 2: 1%) | **3.5%** (6: 7%, 7: 0%) |
| Client 2 | **30.5%** (2: 1%, 3: 60%) | **0.0%** (7: 0%, 8: 0%) |
| Client 3 | **66.0%** (3: 60%, 4: 72%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **36.0%** (4: 72%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) |
