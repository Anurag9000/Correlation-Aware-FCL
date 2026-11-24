# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 4.00% on 26 observed labels.
Validation accuracy: 0.038461538461538464; seen labels never predicted: 17.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.04521739130434783.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **49.0%** (0: 0%, 1: 98%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **49.0%** (1: 98%, 2: 0%) | **1.5%** (6: 0%, 7: 3%) | **0.0%** (11: 0%, 12: 0%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] | **0.0%** (21: 0%, 22: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **2.5%** (7: 3%, 8: 2%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) | **0.5%** (22: 0%, 23: 1%) |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **1.0%** (8: 2%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **0.0%** (18: 0%, 19: 0%) | **0.5%** (23: 1%, 24: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **0.0%** (24: 0%, 25: 0%) |
