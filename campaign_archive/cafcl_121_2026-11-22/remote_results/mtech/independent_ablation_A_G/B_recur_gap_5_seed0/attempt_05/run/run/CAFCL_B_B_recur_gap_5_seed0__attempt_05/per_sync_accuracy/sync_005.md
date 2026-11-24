# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.00% on 50 observed labels.
Validation accuracy: 0.8432; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.84.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 94%, 1: 83%) | **73.5%** (10: 82%, 11: 65%) | **95.5%** (20: 92%, 21: 99%) | **94.0%** (30: 99%, 31: 89%) | **75.5%** (40: 96%, 41: 55%) |
| Client 1 | **89.0%** (2: 82%, 3: 96%) | **73.5%** (12: 94%, 13: 53%) | **86.5%** (22: 75%, 23: 98%) | **87.5%** (32: 85%, 33: 90%) | **78.0%** (42: 93%, 43: 63%) |
| Client 2 | **84.5%** (4: 91%, 5: 78%) | **73.0%** (14: 50%, 15: 96%) | **88.5%** (24: 86%, 25: 91%) | **81.5%** (34: 81%, 35: 82%) | **75.0%** (44: 86%, 45: 64%) |
| Client 3 | **81.5%** (6: 91%, 7: 72%) | **92.5%** (16: 91%, 17: 94%) | **82.0%** (26: 70%, 27: 94%) | **88.0%** (36: 89%, 37: 87%) | **91.5%** (46: 98%, 47: 85%) |
| Client 4 | **85.5%** (8: 84%, 9: 87%) | **77.5%** (18: 74%, 19: 81%) | **81.0%** (28: 91%, 29: 71%) | **84.5%** (38: 86%, 39: 83%) | **92.0%** (48: 87%, 49: 97%) |
