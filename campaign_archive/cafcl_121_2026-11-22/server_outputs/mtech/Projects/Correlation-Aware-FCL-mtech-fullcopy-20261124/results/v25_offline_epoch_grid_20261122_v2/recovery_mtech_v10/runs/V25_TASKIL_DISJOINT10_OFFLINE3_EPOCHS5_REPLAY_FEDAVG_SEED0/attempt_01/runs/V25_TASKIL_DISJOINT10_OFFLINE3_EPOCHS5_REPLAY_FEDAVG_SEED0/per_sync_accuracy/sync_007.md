# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.50% on 70 observed labels.
Validation accuracy: 0.7342857142857143; seen labels never predicted: 2.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.7940384615384616.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **82.0%** (0: 71%, 1: 93%) | **56.0%** (10: 93%, 11: 19%) | **91.5%** (20: 94%, 21: 89%) | **92.5%** (30: 92%, 31: 93%) | **74.0%** (40: 90%, 41: 58%) | **69.5%** (50: 51%, 51: 88%) | **94.0%** (60: 90%, 61: 98%) |
| Client 1 | **83.5%** (2: 84%, 3: 83%) | **69.0%** (12: 85%, 13: 53%) | **89.5%** (22: 89%, 23: 90%) | **50.0%** (32: 97%, 33: 3%) [PENDING upload] | **63.0%** (42: 74%, 43: 52%) [PENDING upload] | **50.0%** (52: 100%, 53: 0%) [PENDING upload] | **69.0%** (62: 96%, 63: 42%) [PENDING upload] |
| Client 2 | **81.0%** (4: 91%, 5: 71%) | **68.5%** (14: 39%, 15: 98%) | **83.0%** (24: 82%, 25: 84%) | **68.5%** (34: 91%, 35: 46%) | **58.0%** (44: 94%, 45: 22%) | **34.5%** (54: 62%, 55: 7%) [PENDING upload] | **50.5%** (64: 12%, 65: 89%) [PENDING upload] |
| Client 3 | **69.0%** (6: 56%, 7: 82%) | **90.0%** (16: 91%, 17: 89%) | **74.5%** (26: 57%, 27: 92%) | **82.0%** (36: 83%, 37: 81%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **46.0%** (56: 16%, 57: 76%) [PENDING upload] | **60.0%** (66: 74%, 67: 46%) [PENDING upload] |
| Client 4 | **84.0%** (8: 81%, 9: 87%) | **82.5%** (18: 80%, 19: 85%) | **78.5%** (28: 71%, 29: 86%) | **79.0%** (38: 84%, 39: 74%) | **93.0%** (48: 94%, 49: 92%) | **83.5%** (58: 73%, 59: 94%) | **88.0%** (68: 83%, 69: 93%) |
