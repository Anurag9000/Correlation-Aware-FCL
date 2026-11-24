# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 7.20% on 30 observed labels.
Validation accuracy: 0.066; seen labels never predicted: 17.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.07714285714285714.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **13.5%** (0: 27%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **5.5%** (2: 0%, 3: 11%) | **19.5%** (12: 38%, 13: 1%) | **0.0%** (22: 0%, 23: 0%) [PENDING upload] |
| Client 2 | **47.5%** (4: 89%, 5: 6%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **18.0%** (6: 5%, 7: 31%) | **1.0%** (16: 2%, 17: 0%) | **3.0%** (26: 6%, 27: 0%) |
| Client 4 | **0.0%** (8: 0%, 9: 0%) | **0.0%** (18: 0%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) |
