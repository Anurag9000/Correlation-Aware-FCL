# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.42% on 48 observed labels.
Validation accuracy: 0.8541666666666666; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.8541666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 86%, 1: 91%) | **74.5%** (10: 87%, 11: 62%) | **94.5%** (20: 89%, 21: 100%) | **88.5%** (0: 86%, 1: 91%) | **91.5%** (40: 93%, 41: 90%) |
| Client 1 | **87.5%** (2: 80%, 3: 95%) | **78.5%** (12: 96%, 13: 61%) | **88.5%** (22: 80%, 23: 97%) | **86.5%** (32: 81%, 33: 92%) | **79.5%** (42: 85%, 43: 74%) |
| Client 2 | **88.0%** (4: 91%, 5: 85%) | **81.5%** (14: 70%, 15: 93%) | **88.0%** (24: 87%, 25: 89%) | **82.5%** (34: 86%, 35: 79%) | **73.5%** (44: 65%, 45: 82%) |
| Client 3 | **81.0%** (6: 82%, 7: 80%) | **93.0%** (16: 88%, 17: 98%) | **80.5%** (26: 64%, 27: 97%) | **89.0%** (36: 89%, 37: 89%) | **92.5%** (46: 96%, 47: 89%) |
| Client 4 | **85.5%** (8: 87%, 9: 84%) | **83.0%** (18: 88%, 19: 78%) | **80.5%** (28: 85%, 29: 76%) | **87.5%** (38: 94%, 39: 81%) | **94.5%** (48: 93%, 49: 96%) |
