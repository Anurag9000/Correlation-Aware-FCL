# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 42.67% on 42 observed labels.
Validation accuracy: 0.4380952380952381; seen labels never predicted: 4.
Server-accepted classes: 42; pending private classes: 0.
Accepted-only TEST accuracy: 0.4266666666666667.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **58.0%** (0: 69%, 1: 47%) | **39.0%** (6: 40%, 7: 38%) | **65.0%** (12: 74%, 13: 56%) | **40.0%** (18: 49%, 19: 31%) | **53.5%** (24: 77%, 25: 30%) | **56.5%** (30: 79%, 31: 34%) | **0.0%** (36: 0%, 37: 0%) |
| Client 1 | **31.0%** (2: 35%, 3: 27%) | **57.5%** (8: 67%, 9: 48%) | **44.0%** (14: 31%, 15: 57%) | **77.5%** (20: 79%, 21: 76%) | **53.5%** (26: 50%, 27: 57%) | **44.0%** (32: 29%, 33: 59%) | **0.0%** (38: 0%, 39: 0%) |
| Client 2 | **33.0%** (4: 30%, 5: 36%) | **33.5%** (10: 33%, 11: 34%) | **68.5%** (16: 60%, 17: 77%) | **55.0%** (22: 36%, 23: 74%) | **49.5%** (28: 49%, 29: 50%) | **36.5%** (34: 37%, 35: 36%) | **0.5%** (40: 1%, 41: 0%) |
