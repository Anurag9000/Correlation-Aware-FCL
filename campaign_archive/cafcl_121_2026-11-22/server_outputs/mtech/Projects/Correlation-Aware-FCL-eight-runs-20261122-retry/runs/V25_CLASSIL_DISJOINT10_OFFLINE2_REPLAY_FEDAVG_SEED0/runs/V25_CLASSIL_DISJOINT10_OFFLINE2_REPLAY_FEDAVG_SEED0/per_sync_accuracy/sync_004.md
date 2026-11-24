# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.57% on 40 observed labels.
Validation accuracy: 0.1115; seen labels never predicted: 23.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.141.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **18.5%** (0: 1%, 1: 36%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) |
| Client 1 | **50.5%** (2: 6%, 3: 95%) | **33.0%** (12: 31%, 13: 35%) | **26.0%** (22: 0%, 23: 52%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] |
| Client 3 | **7.5%** (6: 6%, 7: 9%) | **12.0%** (16: 6%, 17: 18%) | **5.5%** (26: 0%, 27: 11%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **38.0%** (8: 54%, 9: 22%) | **20.5%** (18: 40%, 19: 1%) | **0.0%** (28: 0%, 29: 0%) | **0.0%** (38: 0%, 39: 0%) |
