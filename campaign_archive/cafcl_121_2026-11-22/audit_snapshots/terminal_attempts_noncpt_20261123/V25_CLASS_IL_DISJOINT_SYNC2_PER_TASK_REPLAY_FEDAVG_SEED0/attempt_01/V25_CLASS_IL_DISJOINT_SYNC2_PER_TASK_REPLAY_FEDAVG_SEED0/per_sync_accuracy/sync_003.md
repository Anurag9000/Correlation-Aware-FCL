# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.70% on 20 observed labels.
Validation accuracy: 0.178; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.217.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **55.0%** (0: 65%, 1: 45%) | **55.0%** (0: 65%, 1: 45%) | **0.5%** (10: 1%, 11: 0%) |
| Client 1 | **39.5%** (2: 48%, 3: 31%) | **39.5%** (2: 48%, 3: 31%) | **0.5%** (12: 0%, 13: 1%) |
| Client 2 | **46.0%** (4: 60%, 5: 32%) | **46.0%** (4: 60%, 5: 32%) | **0.5%** (14: 1%, 15: 0%) |
| Client 3 | **29.0%** (6: 22%, 7: 36%) | **29.0%** (6: 22%, 7: 36%) | **0.0%** (16: 0%, 17: 0%) |
| Client 4 | **45.5%** (8: 71%, 9: 20%) | **45.5%** (8: 71%, 9: 20%) | **0.5%** (18: 1%, 19: 0%) |
