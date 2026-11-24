# Server sync 11 — global model v11

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.93% on 44 observed labels.
Validation accuracy: 0.42045454545454547; seen labels never predicted: 0.
Server-accepted classes: 44; pending private classes: 0.
Accepted-only TEST accuracy: 0.4193181818181818.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 | U11 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **52.5%** (0: 72%, 1: 33%) | **19.0%** (4: 3%, 5: 35%) | **43.0%** (8: 49%, 9: 37%) | **55.5%** (12: 57%, 13: 54%) | **52.0%** (16: 55%, 17: 49%) | **74.0%** (20: 73%, 21: 75%) | **43.5%** (24: 67%, 25: 20%) | **49.0%** (28: 60%, 29: 38%) | **44.0%** (32: 36%, 33: 52%) | **53.0%** (36: 50%, 37: 56%) | **19.5%** (40: 6%, 41: 33%) |
| Client 1 | **30.5%** (2: 36%, 3: 25%) | **52.5%** (6: 53%, 7: 52%) | **19.0%** (10: 21%, 11: 17%) | **30.0%** (14: 36%, 15: 24%) | **32.5%** (18: 38%, 19: 27%) | **55.0%** (22: 33%, 23: 77%) | **46.0%** (26: 39%, 27: 53%) | **55.5%** (30: 78%, 31: 33%) | **50.0%** (34: 43%, 35: 57%) | **35.5%** (38: 33%, 39: 38%) | **11.0%** (42: 15%, 43: 7%) |
