# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.82% on 50 observed labels.
Validation accuracy: 0.8092; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8082.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 76%, 1: 91%) | **65.5%** (10: 73%, 11: 58%) | **94.0%** (20: 90%, 21: 98%) | **90.0%** (30: 94%, 31: 86%) | **87.0%** (40: 85%, 41: 89%) |
| Client 1 | **85.5%** (2: 81%, 3: 90%) | **66.5%** (12: 92%, 13: 41%) | **84.5%** (22: 76%, 23: 93%) | **85.5%** (32: 83%, 33: 88%) | **64.0%** (42: 98%, 43: 30%) |
| Client 2 | **82.0%** (4: 86%, 5: 78%) | **75.5%** (14: 62%, 15: 89%) | **85.0%** (24: 88%, 25: 82%) | **69.5%** (34: 93%, 35: 46%) | **75.5%** (44: 84%, 45: 67%) |
| Client 3 | **78.5%** (6: 76%, 7: 81%) | **91.0%** (16: 86%, 17: 96%) | **71.5%** (26: 47%, 27: 96%) | **91.0%** (36: 93%, 37: 89%) | **89.0%** (46: 83%, 47: 95%) |
| Client 4 | **81.0%** (8: 85%, 9: 77%) | **80.0%** (18: 83%, 19: 77%) | **78.0%** (28: 67%, 29: 89%) | **81.5%** (38: 89%, 39: 74%) | **85.5%** (48: 73%, 49: 98%) |
