# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 4.76% on 21 observed labels.
Validation accuracy: 0.045714285714285714; seen labels never predicted: 15.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.05555555555555555.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **49.5%** (0: 0%, 1: 99%) [PENDING upload] | **0.5%** (5: 1%, 6: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] |
| Client 1 | **49.5%** (1: 99%, 2: 0%) | **0.0%** (6: 0%, 7: 0%) | **0.0%** (11: 0%, 12: 0%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **0.0%** (8: 0%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **0.0%** (18: 0%, 19: 0%) |
| Client 4 | **0.5%** (4: 0%, 5: 1%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) |
