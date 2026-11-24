# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.70% on 50 observed labels.
Validation accuracy: 0.8512; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.847.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 92%, 1: 87%) | **77.0%** (10: 86%, 11: 68%) | **96.0%** (20: 94%, 21: 98%) | **94.0%** (30: 95%, 31: 93%) | **84.0%** (40: 98%, 41: 70%) |
| Client 1 | **89.5%** (2: 86%, 3: 93%) | **78.0%** (12: 94%, 13: 62%) | **88.5%** (22: 79%, 23: 98%) | **88.5%** (32: 83%, 33: 94%) | **77.5%** (42: 92%, 43: 63%) |
| Client 2 | **87.0%** (4: 86%, 5: 88%) | **77.5%** (14: 60%, 15: 95%) | **86.0%** (24: 84%, 25: 88%) | **81.5%** (34: 94%, 35: 69%) | **71.5%** (44: 93%, 45: 50%) |
| Client 3 | **81.0%** (6: 76%, 7: 86%) | **92.0%** (16: 89%, 17: 95%) | **81.0%** (26: 70%, 27: 92%) | **88.0%** (36: 88%, 37: 88%) | **89.5%** (46: 100%, 47: 79%) |
| Client 4 | **82.0%** (8: 84%, 9: 80%) | **78.5%** (18: 73%, 19: 84%) | **80.5%** (28: 88%, 29: 73%) | **85.0%** (38: 84%, 39: 86%) | **94.0%** (48: 91%, 49: 97%) |
