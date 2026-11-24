# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.28% on 40 observed labels.
Validation accuracy: 0.063; seen labels never predicted: 27.
Server-accepted classes: 26; pending private classes: 14.
Accepted-only TEST accuracy: 0.09653846153846155.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] | **0.0%** (30: 0%, 31: 0%) [PENDING upload] |
| Client 1 | **10.5%** (2: 0%, 3: 21%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **1.5%** (14: 0%, 15: 3%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **7.5%** (26: 15%, 27: 0%) | **23.5%** (36: 13%, 37: 34%) |
| Client 4 | **41.0%** (8: 73%, 9: 9%) | **40.0%** (18: 73%, 19: 7%) | **0.0%** (28: 0%, 29: 0%) | **1.5%** (38: 0%, 39: 3%) |
