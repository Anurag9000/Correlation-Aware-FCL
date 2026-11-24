# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.31% on 36 observed labels.
Validation accuracy: 0.41555555555555557; seen labels never predicted: 4.
Server-accepted classes: 36; pending private classes: 0.
Accepted-only TEST accuracy: 0.41305555555555556.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 77%, 1: 57%) | **48.0%** (6: 52%, 7: 44%) | **56.0%** (12: 62%, 13: 50%) | **49.5%** (18: 50%, 19: 49%) | **57.5%** (24: 77%, 25: 38%) | **1.5%** (30: 3%, 31: 0%) |
| Client 1 | **40.5%** (2: 41%, 3: 40%) | **43.0%** (8: 55%, 9: 31%) | **53.5%** (14: 49%, 15: 58%) | **70.5%** (20: 73%, 21: 68%) | **34.5%** (26: 30%, 27: 39%) | **0.0%** (32: 0%, 33: 0%) |
| Client 2 | **40.0%** (4: 54%, 5: 26%) | **33.0%** (10: 38%, 11: 28%) | **55.5%** (16: 38%, 17: 73%) | **63.0%** (22: 41%, 23: 85%) | **30.5%** (28: 37%, 29: 24%) | **0.0%** (34: 0%, 35: 0%) |
