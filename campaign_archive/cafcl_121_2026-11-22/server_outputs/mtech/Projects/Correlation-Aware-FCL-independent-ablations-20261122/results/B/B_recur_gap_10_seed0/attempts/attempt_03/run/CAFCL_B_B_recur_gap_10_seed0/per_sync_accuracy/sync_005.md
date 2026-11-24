# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.82% on 50 observed labels.
Validation accuracy: 0.8492; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8482.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.0%** (0: 90%, 1: 90%) | **75.0%** (10: 85%, 11: 65%) | **95.5%** (20: 95%, 21: 96%) | **96.5%** (30: 98%, 31: 95%) | **80.5%** (40: 100%, 41: 61%) |
| Client 1 | **92.0%** (2: 89%, 3: 95%) | **79.5%** (12: 94%, 13: 65%) | **86.0%** (22: 75%, 23: 97%) | **87.5%** (32: 81%, 33: 94%) | **77.5%** (42: 89%, 43: 66%) |
| Client 2 | **87.0%** (4: 91%, 5: 83%) | **75.5%** (14: 53%, 15: 98%) | **86.5%** (24: 86%, 25: 87%) | **82.0%** (34: 90%, 35: 74%) | **77.0%** (44: 89%, 45: 65%) |
| Client 3 | **79.5%** (6: 85%, 7: 74%) | **92.0%** (16: 88%, 17: 96%) | **80.5%** (26: 63%, 27: 98%) | **89.0%** (36: 92%, 37: 86%) | **89.5%** (46: 99%, 47: 80%) |
| Client 4 | **84.5%** (8: 85%, 9: 84%) | **79.5%** (18: 70%, 19: 89%) | **84.0%** (28: 93%, 29: 75%) | **85.0%** (38: 88%, 39: 82%) | **89.0%** (48: 80%, 49: 98%) |
