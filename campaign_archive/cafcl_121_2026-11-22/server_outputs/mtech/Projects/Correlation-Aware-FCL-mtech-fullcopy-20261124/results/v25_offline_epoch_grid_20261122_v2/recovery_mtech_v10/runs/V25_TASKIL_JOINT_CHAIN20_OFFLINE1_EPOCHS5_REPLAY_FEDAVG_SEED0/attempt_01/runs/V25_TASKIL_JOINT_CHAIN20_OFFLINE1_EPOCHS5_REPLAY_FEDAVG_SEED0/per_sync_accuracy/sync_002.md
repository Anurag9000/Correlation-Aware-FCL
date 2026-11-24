# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 57.91% on 11 observed labels.
Validation accuracy: 0.5690909090909091; seen labels never predicted: 0.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5836363636363636.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **79.0%** (0: 86%, 1: 72%) | **46.0%** (5: 40%, 6: 52%) |
| Client 1 | **43.5%** (1: 72%, 2: 15%) | **64.0%** (6: 52%, 7: 76%) |
| Client 2 | **13.0%** (2: 15%, 3: 11%) | **79.0%** (7: 76%, 8: 82%) |
| Client 3 | **51.5%** (3: 11%, 4: 92%) | **46.5%** (8: 82%, 9: 11%) |
| Client 4 | **66.0%** (4: 92%, 5: 40%) | **55.5%** (9: 11%, 10: 100%) |
