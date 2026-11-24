# Server sync 12 — global model v12

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 89.12% on 48 observed labels.
Validation accuracy: 0.8908333333333334; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.89125.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 | U11 | U12 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **91.0%** (0: 90%, 1: 92%) | **90.5%** (4: 88%, 5: 93%) | **89.0%** (8: 92%, 9: 86%) | **83.5%** (12: 86%, 13: 81%) | **96.5%** (16: 96%, 17: 97%) | **97.5%** (20: 96%, 21: 99%) | **92.5%** (24: 88%, 25: 97%) | **87.5%** (28: 93%, 29: 82%) | **90.5%** (32: 87%, 33: 94%) | **92.5%** (36: 90%, 37: 95%) | **94.0%** (40: 91%, 41: 97%) | **78.5%** (44: 94%, 45: 63%) |
| Client 1 | **95.5%** (2: 96%, 3: 95%) | **80.0%** (6: 79%, 7: 81%) | **84.5%** (10: 85%, 11: 84%) | **89.0%** (14: 87%, 15: 91%) | **83.0%** (18: 92%, 19: 74%) | **94.0%** (22: 93%, 23: 95%) | **83.5%** (26: 73%, 27: 94%) | **94.5%** (30: 100%, 31: 89%) | **88.5%** (34: 85%, 35: 92%) | **87.0%** (38: 84%, 39: 90%) | **79.5%** (42: 67%, 43: 92%) | **96.5%** (46: 98%, 47: 95%) |
