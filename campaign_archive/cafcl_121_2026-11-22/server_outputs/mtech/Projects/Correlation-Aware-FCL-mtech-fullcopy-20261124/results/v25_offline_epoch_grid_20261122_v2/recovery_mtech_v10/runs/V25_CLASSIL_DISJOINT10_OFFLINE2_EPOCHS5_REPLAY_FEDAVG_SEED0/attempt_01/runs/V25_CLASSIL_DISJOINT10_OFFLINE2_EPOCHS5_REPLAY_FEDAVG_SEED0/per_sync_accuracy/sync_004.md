# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.75% on 40 observed labels.
Validation accuracy: 0.134; seen labels never predicted: 20.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.18333333333333332.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **45.0%** (0: 48%, 1: 42%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (20: 0%, 21: 0%) | **0.0%** (30: 0%, 31: 0%) |
| Client 1 | **50.0%** (2: 25%, 3: 75%) | **35.0%** (12: 55%, 13: 15%) | **6.5%** (22: 4%, 23: 9%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] | **0.0%** (34: 0%, 35: 0%) [PENDING upload] |
| Client 3 | **15.5%** (6: 14%, 7: 17%) | **40.5%** (16: 42%, 17: 39%) | **8.5%** (26: 11%, 27: 6%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **42.5%** (8: 57%, 9: 28%) | **28.5%** (18: 52%, 19: 5%) | **3.0%** (28: 5%, 29: 1%) | **0.0%** (38: 0%, 39: 0%) |
