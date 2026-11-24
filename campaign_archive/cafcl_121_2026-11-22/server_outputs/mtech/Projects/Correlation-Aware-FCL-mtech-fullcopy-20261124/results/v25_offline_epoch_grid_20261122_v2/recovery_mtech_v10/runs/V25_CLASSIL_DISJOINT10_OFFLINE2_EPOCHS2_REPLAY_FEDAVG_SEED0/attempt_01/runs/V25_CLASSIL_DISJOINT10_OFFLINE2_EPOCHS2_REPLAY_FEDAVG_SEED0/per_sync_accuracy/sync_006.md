# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.95% on 60 observed labels.
Validation accuracy: 0.12533333333333332; seen labels never predicted: 18.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.1765909090909091.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **43.5%** (0: 57%, 1: 30%) | **1.0%** (10: 2%, 11: 0%) | **18.5%** (20: 32%, 21: 5%) | **16.5%** (30: 20%, 31: 13%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] | **0.0%** (50: 0%, 51: 0%) [PENDING upload] |
| Client 1 | **32.5%** (2: 40%, 3: 25%) | **20.0%** (12: 16%, 13: 24%) | **38.5%** (22: 2%, 23: 75%) | **6.0%** (32: 0%, 33: 12%) | **11.5%** (42: 0%, 43: 23%) | **33.0%** (52: 2%, 53: 64%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **10.0%** (6: 16%, 7: 4%) | **30.0%** (16: 2%, 17: 58%) | **27.5%** (26: 4%, 27: 51%) | **17.0%** (36: 27%, 37: 7%) | **8.0%** (46: 8%, 47: 8%) | **5.5%** (56: 0%, 57: 11%) |
| Client 4 | **14.5%** (8: 22%, 9: 7%) | **26.0%** (18: 48%, 19: 4%) | **1.5%** (28: 2%, 29: 1%) | **4.5%** (38: 9%, 39: 0%) | **15.5%** (48: 5%, 49: 26%) | **7.5%** (58: 8%, 59: 7%) |
