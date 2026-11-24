# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.48% on 50 observed labels.
Validation accuracy: 0.7452; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.7510416666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **79.0%** (0: 72%, 1: 86%) | **58.5%** (10: 96%, 11: 21%) | **93.5%** (20: 90%, 21: 97%) | **91.5%** (30: 84%, 31: 99%) | **69.5%** (40: 88%, 41: 51%) |
| Client 1 | **69.0%** (2: 38%, 3: 100%) | **69.0%** (12: 87%, 13: 51%) | **88.0%** (22: 92%, 23: 84%) | **63.0%** (32: 28%, 33: 98%) | **59.0%** (42: 18%, 43: 100%) |
| Client 2 | **73.5%** (4: 96%, 5: 51%) | **70.0%** (14: 48%, 15: 92%) | **81.0%** (24: 66%, 25: 96%) | **57.5%** (34: 98%, 35: 17%) | **65.5%** (44: 94%, 45: 37%) |
| Client 3 | **69.5%** (6: 53%, 7: 86%) | **86.5%** (16: 86%, 17: 87%) | **78.0%** (26: 57%, 27: 99%) | **87.0%** (36: 79%, 37: 95%) | **93.0%** (46: 94%, 47: 92%) |
| Client 4 | **78.0%** (8: 96%, 9: 60%) | **81.0%** (18: 91%, 19: 71%) | **71.5%** (28: 65%, 29: 78%) | **70.5%** (38: 95%, 39: 46%) | **34.5%** (48: 56%, 49: 13%) [PENDING upload] |
