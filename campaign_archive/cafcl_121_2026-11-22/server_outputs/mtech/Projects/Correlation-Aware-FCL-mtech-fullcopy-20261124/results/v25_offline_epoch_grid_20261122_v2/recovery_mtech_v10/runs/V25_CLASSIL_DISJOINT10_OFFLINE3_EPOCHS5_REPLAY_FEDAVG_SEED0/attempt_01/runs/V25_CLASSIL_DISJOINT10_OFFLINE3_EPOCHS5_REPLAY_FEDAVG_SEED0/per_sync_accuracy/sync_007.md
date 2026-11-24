# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.70% on 70 observed labels.
Validation accuracy: 0.18085714285714286; seen labels never predicted: 19.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.23826923076923076.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **28.0%** (0: 18%, 1: 38%) | **0.0%** (10: 0%, 11: 0%) | **21.5%** (20: 37%, 21: 6%) | **28.5%** (30: 50%, 31: 7%) | **15.0%** (40: 0%, 41: 30%) | **3.5%** (50: 0%, 51: 7%) | **27.0%** (60: 12%, 61: 42%) |
| Client 1 | **29.0%** (2: 31%, 3: 27%) | **31.0%** (12: 10%, 13: 52%) | **32.0%** (22: 1%, 23: 63%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] |
| Client 2 | **26.0%** (4: 42%, 5: 10%) | **40.5%** (14: 38%, 15: 43%) | **44.5%** (24: 72%, 25: 17%) | **4.5%** (34: 6%, 35: 3%) | **8.0%** (44: 3%, 45: 13%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] |
| Client 3 | **20.0%** (6: 23%, 7: 17%) | **30.5%** (16: 8%, 17: 53%) | **15.0%** (26: 0%, 27: 30%) | **24.0%** (36: 13%, 37: 35%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] | **0.0%** (66: 0%, 67: 0%) [PENDING upload] |
| Client 4 | **32.0%** (8: 34%, 9: 30%) | **27.5%** (18: 41%, 19: 14%) | **9.0%** (28: 6%, 29: 12%) | **13.5%** (38: 17%, 39: 10%) | **37.0%** (48: 29%, 49: 45%) | **27.0%** (58: 16%, 59: 38%) | **45.0%** (68: 62%, 69: 28%) |
