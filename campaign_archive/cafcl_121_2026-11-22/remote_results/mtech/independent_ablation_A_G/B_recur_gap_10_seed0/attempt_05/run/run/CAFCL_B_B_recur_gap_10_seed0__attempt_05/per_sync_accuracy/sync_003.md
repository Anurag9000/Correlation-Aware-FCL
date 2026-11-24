# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.93% on 30 observed labels.
Validation accuracy: 0.7813333333333333; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.7893333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **84.5%** (0: 71%, 1: 98%) | **64.5%** (10: 92%, 11: 37%) | **89.0%** (20: 78%, 21: 100%) |
| Client 1 | **80.5%** (2: 61%, 3: 100%) | **67.0%** (12: 96%, 13: 38%) | **86.5%** (22: 82%, 23: 91%) |
| Client 2 | **84.0%** (4: 99%, 5: 69%) | **72.5%** (14: 49%, 15: 96%) | **81.5%** (24: 81%, 25: 82%) |
| Client 3 | **77.0%** (6: 69%, 7: 85%) | **86.5%** (16: 75%, 17: 98%) | **80.0%** (26: 64%, 27: 96%) |
| Client 4 | **78.5%** (8: 93%, 9: 64%) | **77.0%** (18: 79%, 19: 75%) | **75.0%** (28: 60%, 29: 90%) |
