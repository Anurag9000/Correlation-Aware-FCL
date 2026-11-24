# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.55% on 40 observed labels.
Validation accuracy: 0.821; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.8155.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 92%, 1: 79%) | **68.5%** (10: 94%, 11: 43%) | **93.5%** (20: 90%, 21: 97%) | **91.5%** (30: 91%, 31: 92%) |
| Client 1 | **89.0%** (2: 80%, 3: 98%) | **71.5%** (12: 93%, 13: 50%) | **89.0%** (22: 80%, 23: 98%) | **80.0%** (32: 69%, 33: 91%) |
| Client 2 | **86.5%** (4: 92%, 5: 81%) | **72.5%** (14: 51%, 15: 94%) | **84.5%** (24: 80%, 25: 89%) | **72.5%** (34: 96%, 35: 49%) |
| Client 3 | **80.5%** (6: 83%, 7: 78%) | **93.0%** (16: 88%, 17: 98%) | **76.5%** (26: 54%, 27: 99%) | **75.0%** (36: 52%, 37: 98%) |
| Client 4 | **81.0%** (8: 87%, 9: 75%) | **81.0%** (18: 80%, 19: 82%) | **81.5%** (28: 76%, 29: 87%) | **78.0%** (38: 73%, 39: 83%) |
