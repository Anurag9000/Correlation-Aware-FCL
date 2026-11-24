# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.25% on 16 observed labels.
Validation accuracy: 0.24625; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.2225.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **40.5%** (0: 71%, 1: 10%) | **34.0%** (5: 50%, 6: 18%) | **0.5%** (10: 0%, 11: 1%) |
| Client 1 | **5.0%** (1: 10%, 2: 0%) | **15.0%** (6: 18%, 7: 12%) | **28.0%** (11: 1%, 12: 55%) |
| Client 2 | **0.5%** (2: 0%, 3: 1%) | **6.0%** (7: 12%, 8: 0%) | **34.0%** (12: 55%, 13: 13%) |
| Client 3 | **27.5%** (3: 1%, 4: 54%) | **0.0%** (8: 0%, 9: 0%) | **29.5%** (13: 13%, 14: 46%) |
| Client 4 | **52.0%** (4: 54%, 5: 50%) | **0.0%** (9: 0%, 10: 0%) | **35.5%** (14: 46%, 15: 25%) |
