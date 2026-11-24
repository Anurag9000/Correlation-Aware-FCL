# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.17% on 30 observed labels.
Validation accuracy: 0.058666666666666666; seen labels never predicted: 23.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.0775.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **3.0%** (2: 6%, 3: 0%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **14.0%** (26: 28%, 27: 0%) |
| Client 4 | **14.0%** (8: 28%, 9: 0%) | **46.5%** (18: 88%, 19: 5%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
