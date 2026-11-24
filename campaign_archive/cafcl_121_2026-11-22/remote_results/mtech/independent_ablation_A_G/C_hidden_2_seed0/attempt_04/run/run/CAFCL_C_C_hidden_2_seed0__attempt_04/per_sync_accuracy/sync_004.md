# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.28% on 40 observed labels.
Validation accuracy: 0.753; seen labels never predicted: 1.
Server-accepted classes: 36; pending private classes: 4.
Accepted-only TEST accuracy: 0.7919444444444445.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **82.5%** (8: 80%, 9: 85%) | **73.5%** (14: 73%, 15: 74%) | **74.5%** (26: 55%, 27: 94%) | **77.0%** (34: 89%, 35: 65%) |
| Client 1 | **65.5%** (6: 45%, 7: 86%) | **78.0%** (18: 67%, 19: 89%) | **74.5%** (28: 82%, 29: 67%) | **89.5%** (30: 98%, 31: 81%) |
| Client 2 | **83.5%** (4: 88%, 5: 79%) | **89.0%** (16: 78%, 17: 100%) | **80.5%** (22: 65%, 23: 96%) | **74.5%** (32: 58%, 33: 91%) |
| Client 3 | **89.0%** (0: 84%, 1: 94%) | **69.5%** (10: 93%, 11: 46%) | **93.5%** (20: 90%, 21: 97%) | **80.0%** (38: 81%, 39: 79%) |
| Client 4 | **84.5%** (2: 74%, 3: 95%) | **66.5%** (12: 97%, 13: 36%) | **50.0%** (24: 100%, 25: 0%) [PENDING upload] | **30.0%** (36: 51%, 37: 9%) [PENDING upload] |
