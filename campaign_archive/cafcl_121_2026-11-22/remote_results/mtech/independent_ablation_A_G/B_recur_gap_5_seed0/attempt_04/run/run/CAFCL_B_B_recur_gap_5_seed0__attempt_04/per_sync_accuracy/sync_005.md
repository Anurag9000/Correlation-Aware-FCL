# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.52% on 50 observed labels.
Validation accuracy: 0.8472; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8352.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 91%, 1: 87%) | **72.5%** (10: 95%, 11: 50%) | **95.5%** (20: 92%, 21: 99%) | **94.0%** (30: 94%, 31: 94%) | **70.5%** (40: 100%, 41: 41%) |
| Client 1 | **90.5%** (2: 83%, 3: 98%) | **78.5%** (12: 91%, 13: 66%) | **85.5%** (22: 73%, 23: 98%) | **84.0%** (32: 71%, 33: 97%) | **74.0%** (42: 88%, 43: 60%) |
| Client 2 | **86.5%** (4: 89%, 5: 84%) | **73.5%** (14: 50%, 15: 97%) | **87.0%** (24: 80%, 25: 94%) | **78.5%** (34: 90%, 35: 67%) | **72.0%** (44: 90%, 45: 54%) |
| Client 3 | **81.5%** (6: 88%, 7: 75%) | **92.0%** (16: 89%, 17: 95%) | **78.5%** (26: 61%, 27: 96%) | **87.0%** (36: 84%, 37: 90%) | **91.5%** (46: 97%, 47: 86%) |
| Client 4 | **83.5%** (8: 84%, 9: 83%) | **80.5%** (18: 75%, 19: 86%) | **85.0%** (28: 91%, 29: 79%) | **83.5%** (38: 85%, 39: 82%) | **93.5%** (48: 90%, 49: 97%) |
