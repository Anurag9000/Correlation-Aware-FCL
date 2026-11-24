# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.52% on 60 observed labels.
Validation accuracy: 0.8193333333333334; seen labels never predicted: 0.
Server-accepted classes: 60; pending private classes: 0.
Accepted-only TEST accuracy: 0.8151666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.7%** (0: 84%, 1: 86%, 2: 81%) | **86.7%** (15: 85%, 16: 85%, 17: 90%) | **85.7%** (30: 91%, 31: 77%, 32: 89%) | **80.0%** (45: 91%, 46: 67%, 47: 82%) |
| Client 1 | **76.7%** (3: 53%, 4: 90%, 5: 87%) | **78.3%** (18: 60%, 19: 87%, 20: 88%) | **85.7%** (33: 89%, 34: 78%, 35: 90%) | **86.0%** (48: 69%, 49: 96%, 50: 93%) |
| Client 2 | **79.0%** (6: 85%, 7: 76%, 8: 76%) | **87.0%** (21: 93%, 22: 75%, 23: 93%) | **82.7%** (36: 89%, 37: 82%, 38: 77%) | **93.7%** (51: 91%, 52: 100%, 53: 90%) |
| Client 3 | **71.3%** (9: 80%, 10: 72%, 11: 62%) | **79.7%** (24: 89%, 25: 82%, 26: 68%) | **83.3%** (39: 84%, 40: 83%, 41: 83%) | **79.3%** (54: 84%, 55: 94%, 56: 60%) |
| Client 4 | **80.0%** (12: 82%, 13: 69%, 14: 89%) | **76.3%** (27: 89%, 28: 76%, 29: 64%) | **71.7%** (42: 71%, 43: 69%, 44: 75%) | **83.7%** (57: 90%, 58: 64%, 59: 97%) |
