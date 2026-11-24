# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.50% on 26 observed labels.
Validation accuracy: 0.12923076923076923; seen labels never predicted: 10.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.135.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 94%, 1: 40%) | **26.0%** (5: 5%, 6: 47%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **20.5%** (1: 40%, 2: 1%) | **24.0%** (6: 47%, 7: 1%) | **33.0%** (11: 0%, 12: 66%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **1.5%** (2: 1%, 3: 2%) | **0.5%** (7: 1%, 8: 0%) | **34.0%** (12: 66%, 13: 2%) | **0.0%** (17: 0%, 18: 0%) | **0.0%** (22: 0%, 23: 0%) |
| Client 3 | **17.0%** (3: 2%, 4: 32%) | **0.0%** (8: 0%, 9: 0%) | **7.5%** (13: 2%, 14: 13%) | **0.0%** (18: 0%, 19: 0%) | **24.0%** (23: 0%, 24: 48%) |
| Client 4 | **18.5%** (4: 32%, 5: 5%) | **0.0%** (9: 0%, 10: 0%) | **6.5%** (14: 13%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **24.0%** (24: 48%, 25: 0%) |
