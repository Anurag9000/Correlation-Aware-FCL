# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.54% on 26 observed labels.
Validation accuracy: 0.11846153846153847; seen labels never predicted: 11.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.12538461538461537.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **68.5%** (0: 53%, 1: 84%) | **19.0%** (5: 10%, 6: 28%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **1.0%** (20: 2%, 21: 0%) |
| Client 1 | **42.0%** (1: 84%, 2: 0%) | **14.5%** (6: 28%, 7: 1%) | **32.0%** (11: 0%, 12: 64%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) | **32.0%** (12: 64%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) | **0.0%** (22: 0%, 23: 0%) |
| Client 3 | **24.0%** (3: 0%, 4: 48%) | **0.0%** (8: 0%, 9: 0%) | **2.0%** (13: 0%, 14: 4%) | **0.0%** (18: 0%, 19: 0%) | **16.0%** (23: 0%, 24: 32%) |
| Client 4 | **29.0%** (4: 48%, 5: 10%) | **0.0%** (9: 0%, 10: 0%) | **2.0%** (14: 4%, 15: 0%) | **1.0%** (19: 0%, 20: 2%) | **16.0%** (24: 32%, 25: 0%) |
