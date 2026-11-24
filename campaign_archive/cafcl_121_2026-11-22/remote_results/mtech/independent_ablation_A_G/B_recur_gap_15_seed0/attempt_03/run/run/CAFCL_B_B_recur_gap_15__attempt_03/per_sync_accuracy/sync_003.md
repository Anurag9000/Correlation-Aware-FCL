# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.53% on 30 observed labels.
Validation accuracy: 0.74; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.7253333333333334.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **72.5%** (0: 45%, 1: 100%) | **57.5%** (10: 95%, 11: 20%) | **87.0%** (20: 74%, 21: 100%) |
| Client 1 | **68.5%** (2: 38%, 3: 99%) | **63.0%** (12: 93%, 13: 33%) | **84.5%** (22: 79%, 23: 90%) |
| Client 2 | **80.0%** (4: 95%, 5: 65%) | **73.5%** (14: 59%, 15: 88%) | **67.5%** (24: 35%, 25: 100%) |
| Client 3 | **62.0%** (6: 31%, 7: 93%) | **82.0%** (16: 65%, 17: 99%) | **69.5%** (26: 40%, 27: 99%) |
| Client 4 | **73.5%** (8: 98%, 9: 49%) | **72.0%** (18: 86%, 19: 58%) | **75.0%** (28: 59%, 29: 91%) |
