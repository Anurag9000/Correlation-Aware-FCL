# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 87.06% on 48 observed labels.
Validation accuracy: 0.87125; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.870625.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.5%** (0: 87%, 1: 88%) | **84.0%** (6: 91%, 7: 77%) | **84.0%** (12: 91%, 13: 77%) | **80.5%** (18: 71%, 19: 90%) | **91.0%** (24: 94%, 25: 88%) | **94.0%** (30: 95%, 31: 93%) | **85.0%** (36: 74%, 37: 96%) | **76.5%** (42: 71%, 43: 82%) |
| Client 1 | **91.5%** (2: 87%, 3: 96%) | **87.5%** (8: 94%, 9: 81%) | **87.5%** (14: 85%, 15: 90%) | **97.0%** (20: 97%, 21: 97%) | **84.5%** (26: 83%, 27: 86%) | **89.0%** (32: 88%, 33: 90%) | **84.5%** (38: 88%, 39: 81%) | **81.0%** (44: 82%, 45: 80%) |
| Client 2 | **88.0%** (4: 95%, 5: 81%) | **79.0%** (10: 78%, 11: 80%) | **95.0%** (16: 91%, 17: 99%) | **92.0%** (22: 88%, 23: 96%) | **83.5%** (28: 85%, 29: 82%) | **79.5%** (34: 83%, 35: 76%) | **92.5%** (40: 92%, 41: 93%) | **95.0%** (46: 95%, 47: 95%) |
