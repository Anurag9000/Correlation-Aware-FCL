# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.12% on 48 observed labels.
Validation accuracy: 0.845; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.84125.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 83%, 1: 95%) | **73.5%** (10: 91%, 11: 56%) | **89.0%** (0: 83%, 1: 95%) | **95.0%** (30: 98%, 31: 92%) | **91.0%** (40: 95%, 41: 87%) |
| Client 1 | **88.0%** (2: 82%, 3: 94%) | **71.5%** (12: 95%, 13: 48%) | **84.5%** (22: 71%, 23: 98%) | **87.0%** (32: 83%, 33: 91%) | **80.0%** (42: 90%, 43: 70%) |
| Client 2 | **88.0%** (4: 91%, 5: 85%) | **73.0%** (14: 56%, 15: 90%) | **85.5%** (24: 84%, 25: 87%) | **80.0%** (34: 85%, 35: 75%) | **74.5%** (44: 72%, 45: 77%) |
| Client 3 | **78.0%** (6: 74%, 7: 82%) | **91.5%** (16: 86%, 17: 97%) | **80.5%** (26: 72%, 27: 89%) | **88.5%** (36: 90%, 37: 87%) | **94.0%** (46: 92%, 47: 96%) |
| Client 4 | **82.0%** (8: 79%, 9: 85%) | **81.5%** (18: 80%, 19: 83%) | **87.5%** (28: 90%, 29: 85%) | **81.5%** (38: 90%, 39: 73%) | **93.5%** (48: 87%, 49: 100%) |
