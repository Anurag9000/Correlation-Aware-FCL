# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.94% on 50 observed labels.
Validation accuracy: 0.8456; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8494.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.0%** (0: 94%, 1: 86%) | **76.0%** (10: 93%, 11: 59%) | **96.0%** (20: 92%, 21: 100%) | **95.5%** (30: 99%, 31: 92%) | **77.5%** (40: 97%, 41: 58%) |
| Client 1 | **92.5%** (2: 87%, 3: 98%) | **75.0%** (12: 92%, 13: 58%) | **85.0%** (22: 72%, 23: 98%) | **85.0%** (32: 74%, 33: 96%) | **77.0%** (42: 91%, 43: 63%) |
| Client 2 | **87.0%** (4: 88%, 5: 86%) | **76.5%** (14: 60%, 15: 93%) | **87.5%** (24: 85%, 25: 90%) | **83.5%** (34: 92%, 35: 75%) | **77.0%** (44: 88%, 45: 66%) |
| Client 3 | **80.0%** (6: 82%, 7: 78%) | **92.0%** (16: 87%, 17: 97%) | **80.5%** (26: 65%, 27: 96%) | **90.0%** (36: 87%, 37: 93%) | **95.0%** (46: 97%, 47: 93%) |
| Client 4 | **85.0%** (8: 86%, 9: 84%) | **81.0%** (18: 82%, 19: 80%) | **82.5%** (28: 89%, 29: 76%) | **86.0%** (38: 87%, 39: 85%) | **90.5%** (48: 84%, 49: 97%) |
