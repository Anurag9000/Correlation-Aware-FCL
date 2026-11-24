# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.47% on 40 observed labels.
Validation accuracy: 0.6635; seen labels never predicted: 1.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.7313333333333333.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **77.0%** (0: 56%, 1: 98%) | **55.0%** (10: 95%, 11: 15%) | **86.5%** (20: 76%, 21: 97%) | **87.5%** (30: 79%, 31: 96%) |
| Client 1 | **85.5%** (2: 73%, 3: 98%) | **57.5%** (12: 92%, 13: 23%) | **72.0%** (22: 51%, 23: 93%) | **50.0%** (32: 100%, 33: 0%) [PENDING upload] |
| Client 2 | **29.0%** (4: 23%, 5: 35%) [PENDING upload] | **48.0%** (14: 73%, 15: 23%) [PENDING upload] | **47.5%** (24: 59%, 25: 36%) [PENDING upload] | **58.0%** (34: 92%, 35: 24%) [PENDING upload] |
| Client 3 | **70.0%** (6: 44%, 7: 96%) | **79.0%** (16: 63%, 17: 95%) | **67.0%** (26: 38%, 27: 96%) | **73.5%** (36: 49%, 37: 98%) |
| Client 4 | **71.5%** (8: 97%, 9: 46%) | **74.5%** (18: 93%, 19: 56%) | **72.0%** (28: 87%, 29: 57%) | **68.5%** (38: 84%, 39: 53%) |
