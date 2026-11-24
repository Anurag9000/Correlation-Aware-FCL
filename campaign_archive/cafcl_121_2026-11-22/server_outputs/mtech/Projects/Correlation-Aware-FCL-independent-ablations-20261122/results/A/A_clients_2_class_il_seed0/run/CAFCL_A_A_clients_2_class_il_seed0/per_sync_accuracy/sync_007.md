# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.71% on 28 observed labels.
Validation accuracy: 0.4307142857142857; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.4471428571428571.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 79%, 1: 55%) | **38.5%** (4: 37%, 5: 40%) | **50.0%** (8: 59%, 9: 41%) | **44.0%** (12: 42%, 13: 46%) | **55.5%** (16: 42%, 17: 69%) | **75.5%** (20: 85%, 21: 66%) | **22.0%** (24: 42%, 25: 2%) |
| Client 1 | **35.0%** (2: 52%, 3: 18%) | **46.5%** (6: 45%, 7: 48%) | **27.5%** (10: 16%, 11: 39%) | **35.0%** (14: 32%, 15: 38%) | **56.0%** (18: 63%, 19: 49%) | **58.0%** (22: 38%, 23: 78%) | **15.5%** (26: 17%, 27: 14%) |
