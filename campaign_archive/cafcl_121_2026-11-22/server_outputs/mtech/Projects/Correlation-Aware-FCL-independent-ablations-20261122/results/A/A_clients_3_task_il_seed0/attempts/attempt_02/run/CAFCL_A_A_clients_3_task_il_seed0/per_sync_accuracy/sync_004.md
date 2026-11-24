# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.33% on 24 observed labels.
Validation accuracy: 0.8216666666666667; seen labels never predicted: 0.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.8233333333333334.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 76%, 1: 92%) | **79.0%** (6: 79%, 7: 79%) | **77.0%** (12: 96%, 13: 58%) | **82.5%** (18: 74%, 19: 91%) |
| Client 1 | **88.0%** (2: 81%, 3: 95%) | **84.5%** (8: 92%, 9: 77%) | **82.0%** (14: 79%, 15: 85%) | **94.5%** (20: 93%, 21: 96%) |
| Client 2 | **83.0%** (4: 88%, 5: 78%) | **69.0%** (10: 94%, 11: 44%) | **90.5%** (16: 83%, 17: 98%) | **74.0%** (22: 49%, 23: 99%) |
