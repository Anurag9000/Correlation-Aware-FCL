# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.10% on 50 observed labels.
Validation accuracy: 0.2316; seen labels never predicted: 12.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.240625.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **44.5%** (0: 46%, 1: 43%) | **25.0%** (10: 28%, 11: 22%) | **59.5%** (20: 56%, 21: 63%) | **0.0%** (30: 0%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) |
| Client 1 | **40.0%** (2: 32%, 3: 48%) | **70.5%** (12: 84%, 13: 57%) | **7.5%** (22: 1%, 23: 14%) | **5.5%** (32: 1%, 33: 10%) | **0.0%** (42: 0%, 43: 0%) |
| Client 2 | **32.0%** (4: 33%, 5: 31%) | **10.5%** (14: 10%, 15: 11%) | **26.0%** (24: 41%, 25: 11%) | **0.0%** (34: 0%, 35: 0%) | **0.0%** (44: 0%, 45: 0%) |
| Client 3 | **44.0%** (6: 34%, 7: 54%) | **61.5%** (16: 49%, 17: 74%) | **29.0%** (26: 10%, 27: 48%) | **5.0%** (36: 10%, 37: 0%) | **0.0%** (46: 0%, 47: 0%) |
| Client 4 | **51.5%** (8: 56%, 9: 47%) | **31.0%** (18: 33%, 19: 29%) | **30.5%** (28: 34%, 29: 27%) | **4.0%** (38: 4%, 39: 4%) | **0.0%** (48: 0%, 49: 0%) [PENDING upload] |
