# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 86.50% on 36 observed labels.
Validation accuracy: 0.8672222222222222; seen labels never predicted: 0.
Server-accepted classes: 36; pending private classes: 0.
Accepted-only TEST accuracy: 0.865.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.0%** (0: 82%, 1: 92%) | **83.0%** (6: 92%, 7: 74%) | **76.5%** (12: 92%, 13: 61%) | **82.5%** (18: 75%, 19: 90%) | **87.5%** (24: 88%, 25: 87%) | **92.0%** (30: 90%, 31: 94%) |
| Client 1 | **90.0%** (2: 83%, 3: 97%) | **86.0%** (8: 96%, 9: 76%) | **87.5%** (14: 91%, 15: 84%) | **95.0%** (20: 92%, 21: 98%) | **83.5%** (26: 84%, 27: 83%) | **87.0%** (32: 78%, 33: 96%) |
| Client 2 | **89.0%** (4: 96%, 5: 82%) | **79.5%** (10: 86%, 11: 73%) | **94.5%** (16: 90%, 17: 99%) | **88.5%** (22: 82%, 23: 95%) | **82.0%** (28: 71%, 29: 93%) | **86.0%** (34: 86%, 35: 86%) |
