# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 64.75% on 40 observed labels.
Validation accuracy: 0.65; seen labels never predicted: 1.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.7083333333333334.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **70.0%** (0: 40%, 1: 100%) | **52.0%** (10: 64%, 11: 40%) | **87.5%** (20: 89%, 21: 86%) | **80.0%** (30: 90%, 31: 70%) |
| Client 1 | **84.0%** (2: 89%, 3: 79%) | **52.0%** (12: 93%, 13: 11%) | **76.0%** (22: 90%, 23: 62%) | **49.5%** (32: 98%, 33: 1%) [PENDING upload] |
| Client 2 | **27.0%** (4: 41%, 5: 13%) [PENDING upload] | **50.0%** (14: 100%, 15: 0%) [PENDING upload] | **53.0%** (24: 93%, 25: 13%) [PENDING upload] | **53.0%** (34: 99%, 35: 7%) [PENDING upload] |
| Client 3 | **58.5%** (6: 57%, 7: 60%) | **76.5%** (16: 71%, 17: 82%) | **71.5%** (26: 56%, 27: 87%) | **72.5%** (36: 76%, 37: 69%) |
| Client 4 | **66.5%** (8: 76%, 9: 57%) | **72.5%** (18: 53%, 19: 92%) | **70.5%** (28: 73%, 29: 68%) | **72.5%** (38: 57%, 39: 88%) |
