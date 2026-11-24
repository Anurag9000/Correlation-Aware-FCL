# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.70% on 40 observed labels.
Validation accuracy: 0.769; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 8.
Accepted-only TEST accuracy: 0.81875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.0%** (8: 96%, 9: 70%) | **69.5%** (14: 59%, 15: 80%) | **77.5%** (26: 68%, 27: 87%) | **79.0%** (34: 93%, 35: 65%) |
| Client 1 | **49.5%** (6: 1%, 7: 98%) [PENDING upload] | **50.5%** (18: 12%, 19: 89%) [PENDING upload] | **51.0%** (28: 46%, 29: 56%) [PENDING upload] | **33.0%** (30: 65%, 31: 1%) [PENDING upload] |
| Client 2 | **87.0%** (4: 97%, 5: 77%) | **91.5%** (16: 84%, 17: 99%) | **84.5%** (22: 75%, 23: 94%) | **78.5%** (32: 64%, 33: 93%) |
| Client 3 | **86.0%** (0: 94%, 1: 78%) | **78.5%** (10: 77%, 11: 80%) | **92.5%** (20: 88%, 21: 97%) | **81.5%** (38: 92%, 39: 71%) |
| Client 4 | **84.0%** (2: 75%, 3: 93%) | **69.5%** (12: 85%, 13: 54%) | **87.0%** (24: 90%, 25: 84%) | **80.5%** (36: 65%, 37: 96%) |
