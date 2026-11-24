# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.57% on 30 observed labels.
Validation accuracy: 0.050666666666666665; seen labels never predicted: 24.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.0835.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **38.0%** (2: 71%, 3: 5%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **1.0%** (26: 2%, 27: 0%) |
| Client 4 | **0.5%** (8: 0%, 9: 1%) | **44.0%** (18: 88%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
