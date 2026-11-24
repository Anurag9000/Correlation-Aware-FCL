# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.60% on 50 observed labels.
Validation accuracy: 0.7272; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.735.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 73%, 1: 89%) | **53.0%** (10: 86%, 11: 20%) | **92.0%** (20: 86%, 21: 98%) | **86.5%** (30: 85%, 31: 88%) | **80.0%** (40: 66%, 41: 94%) |
| Client 1 | **84.0%** (2: 71%, 3: 97%) | **64.0%** (12: 92%, 13: 36%) | **80.5%** (22: 76%, 23: 85%) | **79.0%** (32: 65%, 33: 93%) | **55.0%** (42: 100%, 43: 10%) |
| Client 2 | **80.5%** (4: 87%, 5: 74%) | **64.0%** (14: 34%, 15: 94%) | **70.5%** (24: 69%, 25: 72%) | **57.5%** (34: 99%, 35: 16%) | **67.0%** (44: 90%, 45: 44%) |
| Client 3 | **66.5%** (6: 42%, 7: 91%) | **86.5%** (16: 77%, 17: 96%) | **64.0%** (26: 29%, 27: 99%) | **76.5%** (36: 56%, 37: 97%) | **86.5%** (46: 82%, 47: 91%) |
| Client 4 | **67.5%** (8: 92%, 9: 43%) | **75.5%** (18: 83%, 19: 68%) | **73.0%** (28: 53%, 29: 93%) | **73.5%** (38: 89%, 39: 58%) | **51.0%** (48: 60%, 49: 42%) [PENDING upload] |
