# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.25% on 16 observed labels.
Validation accuracy: 0.11375; seen labels never predicted: 11.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.14.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **42.5%** (0: 0%, 1: 85%) [PENDING upload] | **2.0%** (5: 4%, 6: 0%) | **0.5%** (10: 1%, 11: 0%) |
| Client 1 | **42.5%** (1: 85%, 2: 0%) | **38.5%** (6: 0%, 7: 77%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **38.5%** (7: 77%, 8: 0%) | **0.0%** (12: 0%, 13: 0%) |
| Client 3 | **14.5%** (3: 0%, 4: 29%) | **0.0%** (8: 0%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) |
| Client 4 | **16.5%** (4: 29%, 5: 4%) | **0.5%** (9: 0%, 10: 1%) | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
