# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 3.88% on 40 observed labels.
Validation accuracy: 0.0405; seen labels never predicted: 34.
Server-accepted classes: 26; pending private classes: 14.
Accepted-only TEST accuracy: 0.05961538461538462.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] | **0.0%** (30: 0%, 31: 0%) [PENDING upload] |
| Client 1 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 2 | **11.0%** (4: 22%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **0.5%** (36: 0%, 37: 1%) |
| Client 4 | **48.5%** (8: 97%, 9: 0%) | **16.5%** (18: 33%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) | **1.0%** (38: 0%, 39: 2%) |
