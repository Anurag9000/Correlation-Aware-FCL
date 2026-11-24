# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.30% on 40 observed labels.
Validation accuracy: 0.134; seen labels never predicted: 11.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.15052631578947367.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **39.5%** (0: 33%, 1: 46%) | **0.0%** (10: 0%, 11: 0%) | **3.0%** (20: 1%, 21: 5%) | **0.0%** (30: 0%, 31: 0%) [PENDING upload] |
| Client 1 | **26.0%** (2: 10%, 3: 42%) | **45.0%** (12: 49%, 13: 41%) | **0.0%** (22: 0%, 23: 0%) | **1.0%** (32: 2%, 33: 0%) |
| Client 2 | **40.5%** (4: 66%, 5: 15%) | **3.0%** (14: 5%, 15: 1%) | **0.0%** (24: 0%, 25: 0%) | **0.0%** (34: 0%, 35: 0%) |
| Client 3 | **19.0%** (6: 11%, 7: 27%) | **36.0%** (16: 9%, 17: 63%) | **8.0%** (26: 1%, 27: 15%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **9.5%** (8: 18%, 9: 1%) | **44.0%** (18: 57%, 19: 31%) | **11.5%** (28: 14%, 29: 9%) | **0.0%** (38: 0%, 39: 0%) |
