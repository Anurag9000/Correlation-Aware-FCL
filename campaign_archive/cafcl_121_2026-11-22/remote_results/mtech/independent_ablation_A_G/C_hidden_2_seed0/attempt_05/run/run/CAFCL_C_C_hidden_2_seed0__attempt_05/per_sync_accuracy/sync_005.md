# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.40% on 50 observed labels.
Validation accuracy: 0.7852; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.7927083333333333.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.5%** (8: 86%, 9: 87%) | **76.5%** (14: 80%, 15: 73%) | **74.0%** (26: 50%, 27: 98%) | **81.5%** (34: 88%, 35: 75%) | **91.5%** (48: 85%, 49: 98%) |
| Client 1 | **69.0%** (6: 61%, 7: 77%) | **71.5%** (18: 82%, 19: 61%) | **77.0%** (28: 67%, 29: 87%) | **95.0%** (30: 92%, 31: 98%) | **69.0%** (42: 49%, 43: 89%) |
| Client 2 | **79.5%** (4: 97%, 5: 62%) | **93.5%** (16: 87%, 17: 100%) | **86.0%** (22: 76%, 23: 96%) | **76.5%** (32: 61%, 33: 92%) | **32.5%** (46: 32%, 47: 33%) [PENDING upload] |
| Client 3 | **90.5%** (0: 87%, 1: 94%) | **72.5%** (10: 88%, 11: 57%) | **90.5%** (20: 81%, 21: 100%) | **82.5%** (38: 93%, 39: 72%) | **69.5%** (44: 92%, 45: 47%) |
| Client 4 | **85.5%** (2: 73%, 3: 98%) | **74.5%** (12: 77%, 13: 72%) | **65.5%** (24: 100%, 25: 31%) | **81.0%** (36: 96%, 37: 66%) | **63.5%** (40: 94%, 41: 33%) |
