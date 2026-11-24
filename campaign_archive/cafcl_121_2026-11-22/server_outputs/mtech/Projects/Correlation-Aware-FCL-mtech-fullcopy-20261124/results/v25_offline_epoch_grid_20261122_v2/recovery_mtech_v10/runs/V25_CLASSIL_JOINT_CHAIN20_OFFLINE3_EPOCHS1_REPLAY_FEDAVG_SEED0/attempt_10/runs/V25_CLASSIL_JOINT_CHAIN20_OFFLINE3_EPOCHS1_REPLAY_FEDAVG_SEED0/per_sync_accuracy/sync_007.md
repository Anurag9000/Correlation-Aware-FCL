# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.81% on 36 observed labels.
Validation accuracy: 0.10944444444444444; seen labels never predicted: 24.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.1215625.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **38.5%** (0: 70%, 1: 7%) | **0.5%** (5: 0%, 6: 1%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (25: 0%, 26: 0%) | **45.5%** (30: 86%, 31: 5%) |
| Client 1 | **3.5%** (1: 7%, 2: 0%) | **1.5%** (6: 1%, 7: 2%) | **0.0%** (11: 0%, 12: 0%) | **14.5%** (16: 0%, 17: 29%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] | **2.5%** (31: 5%, 32: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **9.0%** (7: 2%, 8: 16%) | **0.0%** (12: 0%, 13: 0%) | **14.5%** (17: 29%, 18: 0%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **8.0%** (8: 16%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **0.0%** (18: 0%, 19: 0%) | **27.0%** (23: 0%, 24: 54%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] | **35.0%** (33: 0%, 34: 70%) [PENDING upload] |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **27.0%** (24: 54%, 25: 0%) | **43.0%** (29: 0%, 30: 86%) | **59.5%** (34: 70%, 35: 49%) |
