# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.80% on 70 observed labels.
Validation accuracy: 0.22057142857142858; seen labels never predicted: 18.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.29346153846153844.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **25.0%** (0: 44%, 1: 6%) | **10.5%** (10: 21%, 11: 0%) | **36.0%** (20: 46%, 21: 26%) | **44.0%** (30: 66%, 31: 22%) | **12.0%** (40: 6%, 41: 18%) | **11.0%** (50: 10%, 51: 12%) | **26.5%** (60: 48%, 61: 5%) |
| Client 1 | **28.0%** (2: 33%, 3: 23%) | **23.5%** (12: 24%, 13: 23%) | **48.5%** (22: 21%, 23: 76%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] |
| Client 2 | **36.0%** (4: 53%, 5: 19%) | **36.0%** (14: 37%, 15: 35%) | **45.0%** (24: 83%, 25: 7%) | **34.5%** (34: 46%, 35: 23%) | **10.5%** (44: 10%, 45: 11%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] |
| Client 3 | **41.5%** (6: 35%, 7: 48%) | **40.5%** (16: 24%, 17: 57%) | **29.0%** (26: 24%, 27: 34%) | **29.0%** (36: 32%, 37: 26%) | **0.0%** (46: 0%, 47: 0%) [PENDING upload] | **0.0%** (56: 0%, 57: 0%) [PENDING upload] | **0.0%** (66: 0%, 67: 0%) [PENDING upload] |
| Client 4 | **53.0%** (8: 69%, 9: 37%) | **27.0%** (18: 42%, 19: 12%) | **16.0%** (28: 26%, 29: 6%) | **9.0%** (38: 7%, 39: 11%) | **36.0%** (48: 29%, 49: 43%) | **21.5%** (58: 6%, 59: 37%) | **33.5%** (68: 45%, 69: 22%) |
