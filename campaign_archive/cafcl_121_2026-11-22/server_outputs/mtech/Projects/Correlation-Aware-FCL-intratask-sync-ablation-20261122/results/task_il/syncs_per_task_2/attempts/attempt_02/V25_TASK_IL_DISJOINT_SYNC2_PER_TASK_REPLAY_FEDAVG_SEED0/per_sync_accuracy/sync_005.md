# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.60% on 30 observed labels.
Validation accuracy: 0.8273333333333334; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.816.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **85.0%** (0: 82%, 1: 88%) | **85.0%** (0: 82%, 1: 88%) | **68.0%** (10: 90%, 11: 46%) | **68.0%** (10: 90%, 11: 46%) | **93.5%** (20: 88%, 21: 99%) |
| Client 1 | **89.5%** (2: 82%, 3: 97%) | **89.5%** (2: 82%, 3: 97%) | **78.5%** (12: 89%, 13: 68%) | **78.5%** (12: 89%, 13: 68%) | **80.0%** (22: 65%, 23: 95%) |
| Client 2 | **88.0%** (4: 90%, 5: 86%) | **88.0%** (4: 90%, 5: 86%) | **77.0%** (14: 59%, 15: 95%) | **77.0%** (14: 59%, 15: 95%) | **84.0%** (24: 82%, 25: 86%) |
| Client 3 | **80.0%** (6: 74%, 7: 86%) | **80.0%** (6: 74%, 7: 86%) | **92.5%** (16: 90%, 17: 95%) | **92.5%** (16: 90%, 17: 95%) | **64.0%** (26: 93%, 27: 35%) |
| Client 4 | **84.5%** (8: 90%, 9: 79%) | **84.5%** (8: 90%, 9: 79%) | **84.5%** (18: 88%, 19: 81%) | **84.5%** (18: 88%, 19: 81%) | **75.0%** (28: 66%, 29: 84%) |
