# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.29% on 21 observed labels.
Validation accuracy: 0.7257142857142858; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.7328571428571429.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 95%, 1: 76%) | **46.5%** (5: 65%, 6: 28%) | **69.5%** (10: 85%, 11: 54%) | **74.5%** (15: 80%, 16: 69%) |
| Client 1 | **71.5%** (1: 76%, 2: 67%) | **57.5%** (6: 28%, 7: 87%) | **71.0%** (11: 54%, 12: 88%) | **81.5%** (16: 69%, 17: 94%) |
| Client 2 | **80.5%** (2: 67%, 3: 94%) | **89.5%** (7: 87%, 8: 92%) | **64.0%** (12: 88%, 13: 40%) | **92.5%** (17: 94%, 18: 91%) |
| Client 3 | **84.5%** (3: 94%, 4: 75%) | **71.0%** (8: 92%, 9: 50%) | **53.5%** (13: 40%, 14: 67%) | **72.5%** (18: 91%, 19: 54%) |
| Client 4 | **70.0%** (4: 75%, 5: 65%) | **67.5%** (9: 50%, 10: 85%) | **73.5%** (14: 67%, 15: 80%) | **71.0%** (19: 54%, 20: 88%) |
