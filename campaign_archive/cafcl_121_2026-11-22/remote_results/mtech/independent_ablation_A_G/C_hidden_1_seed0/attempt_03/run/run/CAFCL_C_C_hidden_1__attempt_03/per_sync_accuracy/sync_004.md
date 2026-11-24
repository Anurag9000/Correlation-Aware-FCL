# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.12% on 40 observed labels.
Validation accuracy: 0.733; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.7468421052631579.
Terminal flush: False. Selected clients: [0, 1, 2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **71.5%** (8: 98%, 9: 45%) | **67.5%** (14: 43%, 15: 92%) | **61.5%** (26: 23%, 27: 100%) | **68.0%** (34: 99%, 35: 37%) |
| Client 1 | **69.0%** (6: 56%, 7: 82%) | **67.0%** (18: 35%, 19: 99%) | **70.0%** (28: 52%, 29: 88%) | **89.5%** (30: 99%, 31: 80%) |
| Client 2 | **72.5%** (4: 93%, 5: 52%) | **78.0%** (16: 56%, 17: 100%) | **75.5%** (22: 60%, 23: 91%) | **84.0%** (32: 82%, 33: 86%) |
| Client 3 | **84.0%** (0: 72%, 1: 96%) | **74.5%** (10: 87%, 11: 62%) | **91.0%** (20: 82%, 21: 100%) | **43.5%** (38: 64%, 39: 23%) [PENDING upload] |
| Client 4 | **84.0%** (2: 73%, 3: 95%) | **60.5%** (12: 87%, 13: 34%) | **84.0%** (24: 68%, 25: 100%) | **67.0%** (36: 35%, 37: 99%) |
