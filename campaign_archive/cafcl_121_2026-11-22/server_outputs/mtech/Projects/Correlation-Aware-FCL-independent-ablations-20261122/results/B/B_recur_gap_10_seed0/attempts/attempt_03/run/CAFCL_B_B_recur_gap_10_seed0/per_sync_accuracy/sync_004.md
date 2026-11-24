# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.53% on 40 observed labels.
Validation accuracy: 0.8145; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.81525.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **88.0%** (0: 85%, 1: 91%) | **70.5%** (10: 94%, 11: 47%) | **95.0%** (20: 92%, 21: 98%) | **91.5%** (30: 93%, 31: 90%) |
| Client 1 | **85.5%** (2: 73%, 3: 98%) | **65.0%** (12: 90%, 13: 40%) | **87.0%** (22: 78%, 23: 96%) | **81.0%** (32: 73%, 33: 89%) |
| Client 2 | **80.0%** (4: 95%, 5: 65%) | **72.5%** (14: 49%, 15: 96%) | **87.5%** (24: 88%, 25: 87%) | **69.5%** (34: 99%, 35: 40%) |
| Client 3 | **80.0%** (6: 79%, 7: 81%) | **91.0%** (16: 85%, 17: 97%) | **80.0%** (26: 61%, 27: 99%) | **84.0%** (36: 77%, 37: 91%) |
| Client 4 | **82.0%** (8: 89%, 9: 75%) | **79.0%** (18: 77%, 19: 81%) | **79.0%** (28: 71%, 29: 87%) | **82.5%** (38: 84%, 39: 81%) |
