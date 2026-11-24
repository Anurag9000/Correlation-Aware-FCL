# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.97% on 40 observed labels.
Validation accuracy: 0.1475; seen labels never predicted: 13.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.15763157894736843.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **27.0%** (0: 27%, 1: 27%) | **5.0%** (10: 5%, 11: 5%) | **1.0%** (20: 2%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) [PENDING upload] |
| Client 1 | **38.0%** (2: 15%, 3: 61%) | **47.0%** (12: 73%, 13: 21%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (32: 0%, 33: 0%) |
| Client 2 | **50.0%** (4: 66%, 5: 34%) | **1.5%** (14: 3%, 15: 0%) | **0.5%** (24: 1%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) |
| Client 3 | **24.5%** (6: 25%, 7: 24%) | **40.0%** (16: 12%, 17: 68%) | **11.5%** (26: 6%, 27: 17%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **25.5%** (8: 43%, 9: 8%) | **18.5%** (18: 34%, 19: 3%) | **9.5%** (28: 3%, 29: 16%) | **0.0%** (38: 0%, 39: 0%) |
