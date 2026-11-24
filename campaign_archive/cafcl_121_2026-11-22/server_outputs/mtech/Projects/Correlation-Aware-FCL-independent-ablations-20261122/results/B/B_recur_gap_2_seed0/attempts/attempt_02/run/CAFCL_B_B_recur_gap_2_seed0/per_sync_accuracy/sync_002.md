# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.60% on 20 observed labels.
Validation accuracy: 0.596; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.596.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.0%** (0: 16%, 1: 100%) | **53.5%** (10: 99%, 11: 8%) |
| Client 1 | **56.0%** (2: 12%, 3: 100%) | **55.5%** (12: 15%, 13: 96%) |
| Client 2 | **72.5%** (4: 97%, 5: 48%) | **52.5%** (14: 5%, 15: 100%) |
| Client 3 | **58.0%** (6: 24%, 7: 92%) | **76.5%** (16: 62%, 17: 91%) |
| Client 4 | **52.0%** (8: 100%, 9: 4%) | **61.5%** (18: 91%, 19: 32%) |
