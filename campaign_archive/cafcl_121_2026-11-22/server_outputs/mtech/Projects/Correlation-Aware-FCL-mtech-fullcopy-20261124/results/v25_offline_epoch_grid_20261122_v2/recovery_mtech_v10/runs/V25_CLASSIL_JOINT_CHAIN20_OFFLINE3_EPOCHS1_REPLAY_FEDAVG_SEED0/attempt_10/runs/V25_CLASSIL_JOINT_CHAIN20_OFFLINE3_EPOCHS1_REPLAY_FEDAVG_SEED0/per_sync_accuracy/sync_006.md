# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 3.35% on 31 observed labels.
Validation accuracy: 0.032903225806451615; seen labels never predicted: 16.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.03586206896551724.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **48.0%** (0: 0%, 1: 96%) | **0.0%** (5: 0%, 6: 0%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (25: 0%, 26: 0%) |
| Client 1 | **48.0%** (1: 96%, 2: 0%) | **1.5%** (6: 0%, 7: 3%) | **0.0%** (11: 0%, 12: 0%) | **0.5%** (16: 0%, 17: 1%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **3.0%** (7: 3%, 8: 3%) | **0.0%** (12: 0%, 13: 0%) | **1.0%** (17: 1%, 18: 1%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **1.5%** (8: 3%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **0.5%** (18: 1%, 19: 0%) | **0.0%** (23: 0%, 24: 0%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (29: 0%, 30: 0%) |
