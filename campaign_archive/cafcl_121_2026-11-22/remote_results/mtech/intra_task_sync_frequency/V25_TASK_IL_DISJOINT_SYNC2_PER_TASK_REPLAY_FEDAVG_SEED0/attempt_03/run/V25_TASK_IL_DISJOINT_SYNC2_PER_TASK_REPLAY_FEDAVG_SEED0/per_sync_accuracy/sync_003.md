# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.90% on 20 observed labels.
Validation accuracy: 0.693; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.699.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 75%, 1: 92%) | **83.5%** (0: 75%, 1: 92%) | **57.5%** (10: 74%, 11: 41%) |
| Client 1 | **77.5%** (2: 55%, 3: 100%) | **77.5%** (2: 55%, 3: 100%) | **54.5%** (12: 97%, 13: 12%) |
| Client 2 | **79.5%** (4: 92%, 5: 67%) | **79.5%** (4: 92%, 5: 67%) | **61.5%** (14: 41%, 15: 82%) |
| Client 3 | **73.5%** (6: 61%, 7: 86%) | **73.5%** (6: 61%, 7: 86%) | **83.5%** (16: 73%, 17: 94%) |
| Client 4 | **67.5%** (8: 97%, 9: 38%) | **67.5%** (8: 97%, 9: 38%) | **60.5%** (18: 89%, 19: 32%) |
