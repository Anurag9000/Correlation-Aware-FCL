# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.56% on 50 observed labels.
Validation accuracy: 0.1596; seen labels never predicted: 24.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.21789473684210525.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **30.5%** (0: 27%, 1: 34%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **6.0%** (30: 12%, 31: 0%) | **0.0%** (40: 0%, 41: 0%) [PENDING upload] |
| Client 1 | **54.0%** (2: 22%, 3: 86%) | **53.5%** (12: 40%, 13: 67%) | **38.0%** (22: 15%, 23: 61%) | **0.0%** (32: 0%, 33: 0%) | **0.0%** (42: 0%, 43: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] | **0.0%** (44: 0%, 45: 0%) [PENDING upload] |
| Client 3 | **42.0%** (6: 42%, 7: 42%) | **35.5%** (16: 17%, 17: 54%) | **42.5%** (26: 25%, 27: 60%) | **1.5%** (36: 3%, 37: 0%) | **0.5%** (46: 1%, 47: 0%) |
| Client 4 | **60.0%** (8: 76%, 9: 44%) | **36.0%** (18: 44%, 19: 28%) | **13.5%** (28: 19%, 29: 8%) | **0.5%** (38: 0%, 39: 1%) | **0.0%** (48: 0%, 49: 0%) |
