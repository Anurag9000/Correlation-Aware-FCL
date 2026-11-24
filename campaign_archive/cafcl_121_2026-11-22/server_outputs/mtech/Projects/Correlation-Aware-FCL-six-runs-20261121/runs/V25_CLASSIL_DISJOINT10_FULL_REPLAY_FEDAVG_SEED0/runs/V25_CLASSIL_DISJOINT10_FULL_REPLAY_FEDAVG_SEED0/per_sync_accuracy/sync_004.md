# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.95% on 40 observed labels.
Validation accuracy: 0.2965; seen labels never predicted: 9.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.2895.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **52.5%** (0: 67%, 1: 38%) | **14.0%** (10: 8%, 11: 20%) | **56.0%** (20: 58%, 21: 54%) | **0.0%** (30: 0%, 31: 0%) |
| Client 1 | **39.5%** (2: 39%, 3: 40%) | **34.5%** (12: 30%, 13: 39%) | **47.5%** (22: 20%, 23: 75%) | **0.5%** (32: 1%, 33: 0%) |
| Client 2 | **38.0%** (4: 42%, 5: 34%) | **30.5%** (14: 27%, 15: 34%) | **38.0%** (24: 66%, 25: 10%) | **0.0%** (34: 0%, 35: 0%) |
| Client 3 | **40.0%** (6: 42%, 7: 38%) | **45.5%** (16: 12%, 17: 79%) | **35.5%** (26: 20%, 27: 51%) | **0.0%** (36: 0%, 37: 0%) |
| Client 4 | **45.5%** (8: 46%, 9: 45%) | **27.5%** (18: 41%, 19: 14%) | **34.0%** (28: 37%, 29: 31%) | **0.0%** (38: 0%, 39: 0%) |
