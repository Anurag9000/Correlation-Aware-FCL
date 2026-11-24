# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.50% on 16 observed labels.
Validation accuracy: 0.13625; seen labels never predicted: 7.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.1657142857142857.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **17.5%** (0: 0%, 1: 35%) [PENDING upload] | **13.5%** (5: 27%, 6: 0%) | **30.5%** (10: 61%, 11: 0%) |
| Client 1 | **37.5%** (1: 35%, 2: 40%) | **0.5%** (6: 0%, 7: 1%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **20.0%** (2: 40%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) | **0.0%** (12: 0%, 13: 0%) |
| Client 3 | **34.0%** (3: 0%, 4: 68%) | **0.0%** (8: 0%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) |
| Client 4 | **47.5%** (4: 68%, 5: 27%) | **30.5%** (9: 0%, 10: 61%) | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
