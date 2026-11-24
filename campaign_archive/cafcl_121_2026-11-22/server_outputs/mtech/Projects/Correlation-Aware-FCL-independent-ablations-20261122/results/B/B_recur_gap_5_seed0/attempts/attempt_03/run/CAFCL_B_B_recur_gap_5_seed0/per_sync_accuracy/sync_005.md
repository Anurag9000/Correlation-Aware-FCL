# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.44% on 50 observed labels.
Validation accuracy: 0.8556; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8444.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.0%** (0: 92%, 1: 88%) | **73.0%** (10: 89%, 11: 57%) | **96.0%** (20: 92%, 21: 100%) | **93.0%** (30: 97%, 31: 89%) | **80.0%** (40: 97%, 41: 63%) |
| Client 1 | **87.0%** (2: 80%, 3: 94%) | **79.0%** (12: 89%, 13: 69%) | **87.5%** (22: 77%, 23: 98%) | **87.0%** (32: 78%, 33: 96%) | **76.5%** (42: 87%, 43: 66%) |
| Client 2 | **86.0%** (4: 89%, 5: 83%) | **72.0%** (14: 51%, 15: 93%) | **88.0%** (24: 87%, 25: 89%) | **78.0%** (34: 91%, 35: 65%) | **74.0%** (44: 92%, 45: 56%) |
| Client 3 | **85.0%** (6: 81%, 7: 89%) | **92.5%** (16: 90%, 17: 95%) | **77.5%** (26: 61%, 27: 94%) | **88.0%** (36: 88%, 37: 88%) | **93.5%** (46: 96%, 47: 91%) |
| Client 4 | **85.0%** (8: 89%, 9: 81%) | **81.0%** (18: 80%, 19: 82%) | **82.5%** (28: 92%, 29: 73%) | **86.0%** (38: 89%, 39: 83%) | **93.0%** (48: 88%, 49: 98%) |
