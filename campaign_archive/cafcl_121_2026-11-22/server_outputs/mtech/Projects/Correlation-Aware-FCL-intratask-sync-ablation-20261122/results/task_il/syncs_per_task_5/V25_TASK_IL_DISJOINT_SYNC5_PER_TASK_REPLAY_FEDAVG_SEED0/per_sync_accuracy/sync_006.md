# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.80% on 20 observed labels.
Validation accuracy: 0.73; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.738.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.5%** (0: 91%, 1: 90%) | **90.5%** (0: 91%, 1: 90%) | **90.5%** (0: 91%, 1: 90%) | **90.5%** (0: 91%, 1: 90%) | **90.5%** (0: 91%, 1: 90%) | **64.0%** (10: 68%, 11: 60%) |
| Client 1 | **89.5%** (2: 87%, 3: 92%) | **89.5%** (2: 87%, 3: 92%) | **89.5%** (2: 87%, 3: 92%) | **89.5%** (2: 87%, 3: 92%) | **89.5%** (2: 87%, 3: 92%) | **63.0%** (12: 86%, 13: 40%) |
| Client 2 | **84.0%** (4: 92%, 5: 76%) | **84.0%** (4: 92%, 5: 76%) | **84.0%** (4: 92%, 5: 76%) | **84.0%** (4: 92%, 5: 76%) | **84.0%** (4: 92%, 5: 76%) | **50.0%** (14: 100%, 15: 0%) |
| Client 3 | **77.5%** (6: 68%, 7: 87%) | **77.5%** (6: 68%, 7: 87%) | **77.5%** (6: 68%, 7: 87%) | **77.5%** (6: 68%, 7: 87%) | **77.5%** (6: 68%, 7: 87%) | **81.0%** (16: 79%, 17: 83%) |
| Client 4 | **80.0%** (8: 86%, 9: 74%) | **80.0%** (8: 86%, 9: 74%) | **80.0%** (8: 86%, 9: 74%) | **80.0%** (8: 86%, 9: 74%) | **80.0%** (8: 86%, 9: 74%) | **58.5%** (18: 42%, 19: 75%) |
