# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.95% on 20 observed labels.
Validation accuracy: 0.779; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.7695.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 77%, 1: 85%) | **81.0%** (0: 77%, 1: 85%) | **66.0%** (10: 79%, 11: 53%) | **66.0%** (10: 79%, 11: 53%) |
| Client 1 | **82.5%** (2: 66%, 3: 99%) | **82.5%** (2: 66%, 3: 99%) | **68.5%** (12: 99%, 13: 38%) | **68.5%** (12: 99%, 13: 38%) |
| Client 2 | **81.5%** (4: 89%, 5: 74%) | **81.5%** (4: 89%, 5: 74%) | **75.5%** (14: 55%, 15: 96%) | **75.5%** (14: 55%, 15: 96%) |
| Client 3 | **77.0%** (6: 66%, 7: 88%) | **77.0%** (6: 66%, 7: 88%) | **89.5%** (16: 80%, 17: 99%) | **89.5%** (16: 80%, 17: 99%) |
| Client 4 | **77.0%** (8: 95%, 9: 59%) | **77.0%** (8: 95%, 9: 59%) | **71.0%** (18: 44%, 19: 98%) | **71.0%** (18: 44%, 19: 98%) |
