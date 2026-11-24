# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.00% on 31 observed labels.
Validation accuracy: 0.14516129032258066; seen labels never predicted: 14.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.17103448275862068.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **9.0%** (0: 0%, 1: 18%) | **34.5%** (5: 69%, 6: 0%) | **10.5%** (10: 21%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **12.5%** (20: 25%, 21: 0%) | **0.0%** (25: 0%, 26: 0%) |
| Client 1 | **27.0%** (1: 18%, 2: 36%) | **20.5%** (6: 0%, 7: 41%) | **3.5%** (11: 0%, 12: 7%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] |
| Client 2 | **40.5%** (2: 36%, 3: 45%) | **47.5%** (7: 41%, 8: 54%) | **4.0%** (12: 7%, 13: 1%) | **20.5%** (17: 0%, 18: 41%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] |
| Client 3 | **59.5%** (3: 45%, 4: 74%) | **49.0%** (8: 54%, 9: 44%) | **1.0%** (13: 1%, 14: 1%) | **30.0%** (18: 41%, 19: 19%) | **0.0%** (23: 0%, 24: 0%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
| Client 4 | **71.5%** (4: 74%, 5: 69%) | **32.5%** (9: 44%, 10: 21%) | **0.5%** (14: 1%, 15: 0%) | **22.0%** (19: 19%, 20: 25%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (29: 0%, 30: 0%) |
