# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 68.10% on 50 observed labels.
Validation accuracy: 0.6828; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.7494736842105263.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **76.0%** (0: 52%, 1: 100%) | **60.5%** (10: 52%, 11: 69%) | **83.0%** (20: 68%, 21: 98%) | **87.5%** (30: 84%, 31: 91%) | **42.0%** (40: 45%, 41: 39%) [PENDING upload] |
| Client 1 | **86.5%** (2: 82%, 3: 91%) | **61.5%** (12: 92%, 13: 31%) | **81.5%** (22: 76%, 23: 87%) | **80.5%** (32: 77%, 33: 84%) | **70.5%** (42: 95%, 43: 46%) |
| Client 2 | **32.5%** (4: 47%, 5: 18%) [PENDING upload] | **51.5%** (14: 100%, 15: 3%) [PENDING upload] | **52.0%** (24: 96%, 25: 8%) [PENDING upload] | **50.5%** (34: 100%, 35: 1%) [PENDING upload] | **50.0%** (44: 1%, 45: 99%) [PENDING upload] |
| Client 3 | **63.0%** (6: 55%, 7: 71%) | **80.5%** (16: 71%, 17: 90%) | **72.5%** (26: 51%, 27: 94%) | **81.5%** (36: 79%, 37: 84%) | **74.5%** (46: 97%, 47: 52%) |
| Client 4 | **70.5%** (8: 86%, 9: 55%) | **71.0%** (18: 55%, 19: 87%) | **68.5%** (28: 83%, 29: 54%) | **74.0%** (38: 71%, 39: 77%) | **80.5%** (48: 66%, 49: 95%) |
