# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.52% on 21 observed labels.
Validation accuracy: 0.66; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.6666666666666666.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **86.5%** (0: 83%, 1: 90%) | **43.0%** (5: 42%, 6: 44%) | **60.0%** (10: 87%, 11: 33%) | **77.5%** (15: 96%, 16: 59%) |
| Client 1 | **66.0%** (1: 90%, 2: 42%) | **56.0%** (6: 44%, 7: 68%) | **66.0%** (11: 33%, 12: 99%) | **76.5%** (16: 59%, 17: 94%) |
| Client 2 | **59.0%** (2: 42%, 3: 76%) | **82.5%** (7: 68%, 8: 97%) | **52.5%** (12: 99%, 13: 6%) | **90.0%** (17: 94%, 18: 86%) |
| Client 3 | **83.5%** (3: 76%, 4: 91%) | **52.0%** (8: 97%, 9: 7%) | **20.5%** (13: 6%, 14: 35%) | **91.5%** (18: 86%, 19: 97%) |
| Client 4 | **66.5%** (4: 91%, 5: 42%) | **47.0%** (9: 7%, 10: 87%) | **65.5%** (14: 35%, 15: 96%) | **81.0%** (19: 97%, 20: 65%) |
