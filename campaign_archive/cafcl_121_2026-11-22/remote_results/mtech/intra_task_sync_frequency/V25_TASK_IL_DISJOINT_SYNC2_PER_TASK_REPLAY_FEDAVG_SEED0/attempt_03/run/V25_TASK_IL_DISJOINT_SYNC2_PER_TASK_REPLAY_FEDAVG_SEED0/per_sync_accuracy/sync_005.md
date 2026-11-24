# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.17% on 30 observed labels.
Validation accuracy: 0.8266666666666667; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.8116666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 83%, 1: 83%) | **83.0%** (0: 83%, 1: 83%) | **68.0%** (10: 87%, 11: 49%) | **68.0%** (10: 87%, 11: 49%) | **91.5%** (20: 85%, 21: 98%) |
| Client 1 | **81.5%** (2: 66%, 3: 97%) | **81.5%** (2: 66%, 3: 97%) | **78.0%** (12: 92%, 13: 64%) | **78.0%** (12: 92%, 13: 64%) | **87.5%** (22: 82%, 23: 93%) |
| Client 2 | **86.0%** (4: 95%, 5: 77%) | **86.0%** (4: 95%, 5: 77%) | **77.5%** (14: 65%, 15: 90%) | **77.5%** (14: 65%, 15: 90%) | **84.0%** (24: 78%, 25: 90%) |
| Client 3 | **80.5%** (6: 80%, 7: 81%) | **80.5%** (6: 80%, 7: 81%) | **90.5%** (16: 85%, 17: 96%) | **90.5%** (16: 85%, 17: 96%) | **67.5%** (26: 91%, 27: 44%) |
| Client 4 | **85.5%** (8: 90%, 9: 81%) | **85.5%** (8: 90%, 9: 81%) | **82.5%** (18: 84%, 19: 81%) | **82.5%** (18: 84%, 19: 81%) | **74.0%** (28: 52%, 29: 96%) |
