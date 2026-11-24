# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.26% on 50 observed labels.
Validation accuracy: 0.0716; seen labels never predicted: 33.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.09205882352941176.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] | **0.0%** (30: 0%, 31: 0%) [PENDING upload] | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **31.5%** (2: 13%, 3: 50%) | **1.5%** (12: 3%, 13: 0%) | **18.0%** (22: 0%, 23: 36%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **3.0%** (14: 6%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) | **9.0%** (44: 0%, 45: 18%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (26: 0%, 27: 0%) | **20.5%** (36: 33%, 37: 8%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] |
| Client 4 | **35.5%** (8: 71%, 9: 0%) | **30.0%** (18: 60%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) | **3.5%** (38: 1%, 39: 6%) | **4.0%** (48: 5%, 49: 3%) |
