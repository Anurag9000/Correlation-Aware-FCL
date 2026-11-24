# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.93% on 60 observed labels.
Validation accuracy: 0.814; seen labels never predicted: 0.
Server-accepted classes: 56; pending private classes: 4.
Accepted-only TEST accuracy: 0.8364285714285714.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (8: 82%, 9: 86%) | **76.5%** (14: 86%, 15: 67%) | **84.0%** (26: 76%, 27: 92%) | **75.0%** (34: 92%, 35: 58%) | **92.0%** (48: 87%, 49: 97%) | **88.0%** (56: 96%, 57: 80%) |
| Client 1 | **75.0%** (6: 65%, 7: 85%) | **79.0%** (18: 70%, 19: 88%) | **78.0%** (28: 84%, 29: 72%) | **94.5%** (30: 99%, 31: 90%) | **77.0%** (42: 85%, 43: 69%) | **92.0%** (54: 93%, 55: 91%) |
| Client 2 | **86.0%** (4: 86%, 5: 86%) | **96.5%** (16: 97%, 17: 96%) | **88.5%** (22: 81%, 23: 96%) | **78.0%** (32: 69%, 33: 87%) | **36.0%** (46: 27%, 47: 45%) [PENDING upload] | **50.0%** (58: 1%, 59: 99%) [PENDING upload] |
| Client 3 | **83.0%** (0: 82%, 1: 84%) | **81.5%** (10: 87%, 11: 76%) | **92.5%** (20: 86%, 21: 99%) | **83.5%** (38: 92%, 39: 75%) | **72.0%** (44: 81%, 45: 63%) | **78.5%** (50: 67%, 51: 90%) |
| Client 4 | **88.0%** (2: 82%, 3: 94%) | **68.5%** (12: 83%, 13: 54%) | **86.0%** (24: 94%, 25: 78%) | **89.5%** (36: 94%, 37: 85%) | **76.5%** (40: 58%, 41: 95%) | **98.5%** (52: 100%, 53: 97%) |
