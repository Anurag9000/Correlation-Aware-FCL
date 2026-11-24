# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.00% on 16 observed labels.
Validation accuracy: 0.5025; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.53.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **76.0%** (0: 80%, 1: 72%) | **51.0%** (5: 62%, 6: 40%) | **58.5%** (10: 100%, 11: 17%) |
| Client 1 | **67.0%** (1: 72%, 2: 62%) | **32.5%** (6: 40%, 7: 25%) | **57.0%** (11: 17%, 12: 97%) |
| Client 2 | **80.0%** (2: 62%, 3: 98%) | **52.5%** (7: 25%, 8: 80%) | **51.5%** (12: 97%, 13: 6%) |
| Client 3 | **67.0%** (3: 98%, 4: 36%) | **40.0%** (8: 80%, 9: 0%) | **21.0%** (13: 6%, 14: 36%) |
| Client 4 | **49.0%** (4: 36%, 5: 62%) | **50.0%** (9: 0%, 10: 100%) | **36.5%** (14: 36%, 15: 37%) |
