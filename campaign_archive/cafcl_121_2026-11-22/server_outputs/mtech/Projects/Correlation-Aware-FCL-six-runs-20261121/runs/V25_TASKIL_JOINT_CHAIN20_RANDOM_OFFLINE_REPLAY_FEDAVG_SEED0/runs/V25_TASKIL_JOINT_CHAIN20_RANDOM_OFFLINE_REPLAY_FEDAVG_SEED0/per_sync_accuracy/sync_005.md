# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.35% on 26 observed labels.
Validation accuracy: 0.7284615384615385; seen labels never predicted: 0.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.7452.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.5%** (0: 85%, 1: 90%) | **60.5%** (5: 55%, 6: 66%) | **65.0%** (10: 82%, 11: 48%) | **80.0%** (15: 90%, 16: 70%) | **86.5%** (20: 77%, 21: 96%) |
| Client 1 | **80.5%** (1: 90%, 2: 71%) | **59.5%** (6: 66%, 7: 53%) | **71.5%** (11: 48%, 12: 95%) | **84.5%** (16: 70%, 17: 99%) | **81.0%** (21: 96%, 22: 66%) |
| Client 2 | **70.0%** (2: 71%, 3: 69%) | **73.0%** (7: 53%, 8: 93%) | **69.5%** (12: 95%, 13: 44%) | **87.0%** (17: 99%, 18: 75%) | **79.5%** (22: 66%, 23: 93%) |
| Client 3 | **81.5%** (3: 69%, 4: 94%) | **65.5%** (8: 93%, 9: 38%) | **44.0%** (13: 44%, 14: 44%) | **85.0%** (18: 75%, 19: 95%) | **84.0%** (23: 93%, 24: 75%) |
| Client 4 | **74.5%** (4: 94%, 5: 55%) | **60.0%** (9: 38%, 10: 82%) | **67.0%** (14: 44%, 15: 90%) | **86.0%** (19: 95%, 20: 77%) | **59.5%** (24: 75%, 25: 44%) [PENDING upload] |
