# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.56% on 16 observed labels.
Validation accuracy: 0.07125; seen labels never predicted: 9.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.075.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **46.0%** (0: 0%, 1: 92%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) | **0.0%** (10: 0%, 11: 0%) |
| Client 1 | **46.0%** (1: 92%, 2: 0%) | **0.5%** (6: 0%, 7: 1%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) | **0.0%** (12: 0%, 13: 0%) |
| Client 3 | **1.5%** (3: 0%, 4: 3%) | **4.5%** (8: 0%, 9: 9%) | **0.0%** (13: 0%, 14: 0%) |
| Client 4 | **1.5%** (4: 3%, 5: 0%) | **4.5%** (9: 9%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
