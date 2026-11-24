# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.05% on 20 observed labels.
Validation accuracy: 0.772; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.7705.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 86%, 1: 82%) | **84.0%** (0: 86%, 1: 82%) | **64.5%** (10: 76%, 11: 53%) | **64.5%** (10: 76%, 11: 53%) |
| Client 1 | **77.0%** (2: 55%, 3: 99%) | **77.0%** (2: 55%, 3: 99%) | **65.5%** (12: 99%, 13: 32%) | **65.5%** (12: 99%, 13: 32%) |
| Client 2 | **81.0%** (4: 86%, 5: 76%) | **81.0%** (4: 86%, 5: 76%) | **82.5%** (14: 78%, 15: 87%) | **82.5%** (14: 78%, 15: 87%) |
| Client 3 | **76.5%** (6: 65%, 7: 88%) | **76.5%** (6: 65%, 7: 88%) | **86.5%** (16: 79%, 17: 94%) | **86.5%** (16: 79%, 17: 94%) |
| Client 4 | **76.5%** (8: 84%, 9: 69%) | **76.5%** (8: 84%, 9: 69%) | **76.5%** (18: 59%, 19: 94%) | **76.5%** (18: 59%, 19: 94%) |
