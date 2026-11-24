# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.53% on 40 observed labels.
Validation accuracy: 0.0945; seen labels never predicted: 19.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.14033333333333334.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **48.5%** (0: 62%, 1: 35%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) |
| Client 1 | **40.5%** (2: 28%, 3: 53%) | **18.0%** (12: 36%, 13: 0%) | **17.0%** (22: 10%, 23: 24%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] |
| Client 3 | **7.5%** (6: 14%, 7: 1%) | **19.5%** (16: 20%, 17: 19%) | **13.0%** (26: 21%, 27: 5%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **41.5%** (8: 56%, 9: 27%) | **3.0%** (18: 4%, 19: 2%) | **1.5%** (28: 0%, 29: 3%) | **0.5%** (38: 0%, 39: 1%) |
