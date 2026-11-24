# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.88% on 60 observed labels.
Validation accuracy: 0.7703333333333333; seen labels never predicted: 0.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.8584090909090909.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 94%, 1: 85%) | **69.5%** (10: 75%, 11: 64%) | **93.5%** (20: 90%, 21: 97%) | **94.0%** (30: 93%, 31: 95%) | **50.0%** (40: 32%, 41: 68%) [PENDING upload] | **49.0%** (50: 37%, 51: 61%) [PENDING upload] |
| Client 1 | **85.0%** (2: 80%, 3: 90%) | **75.0%** (12: 91%, 13: 59%) | **88.0%** (22: 83%, 23: 93%) | **91.5%** (32: 90%, 33: 93%) | **75.5%** (42: 89%, 43: 62%) | **100.0%** (52: 100%, 53: 100%) |
| Client 2 | **39.0%** (4: 46%, 5: 32%) [PENDING upload] | **47.0%** (14: 84%, 15: 10%) [PENDING upload] | **49.5%** (24: 65%, 25: 34%) [PENDING upload] | **57.5%** (34: 82%, 35: 33%) [PENDING upload] | **53.5%** (44: 17%, 45: 90%) [PENDING upload] | **42.5%** (54: 80%, 55: 5%) [PENDING upload] |
| Client 3 | **81.0%** (6: 84%, 7: 78%) | **85.5%** (16: 81%, 17: 90%) | **79.0%** (26: 60%, 27: 98%) | **90.5%** (36: 92%, 37: 89%) | **94.0%** (46: 91%, 47: 97%) | **89.5%** (56: 91%, 57: 88%) |
| Client 4 | **84.0%** (8: 85%, 9: 83%) | **80.5%** (18: 81%, 19: 80%) | **78.5%** (28: 74%, 29: 83%) | **81.0%** (38: 93%, 39: 69%) | **95.5%** (48: 96%, 49: 95%) | **88.0%** (58: 80%, 59: 96%) |
