# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.96% on 50 observed labels.
Validation accuracy: 0.7872; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.7910416666666666.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.5%** (8: 90%, 9: 85%) | **73.0%** (14: 64%, 15: 82%) | **74.0%** (26: 51%, 27: 97%) | **79.0%** (34: 80%, 35: 78%) | **87.0%** (48: 76%, 49: 98%) |
| Client 1 | **65.5%** (6: 40%, 7: 91%) | **79.5%** (18: 84%, 19: 75%) | **76.0%** (28: 80%, 29: 72%) | **95.0%** (30: 94%, 31: 96%) | **73.5%** (42: 71%, 43: 76%) |
| Client 2 | **82.0%** (4: 97%, 5: 67%) | **91.0%** (16: 82%, 17: 100%) | **81.5%** (22: 67%, 23: 96%) | **74.5%** (32: 53%, 33: 96%) | **50.5%** (46: 30%, 47: 71%) [PENDING upload] |
| Client 3 | **87.0%** (0: 80%, 1: 94%) | **73.0%** (10: 90%, 11: 56%) | **90.5%** (20: 81%, 21: 100%) | **81.0%** (38: 94%, 39: 68%) | **65.0%** (44: 97%, 45: 33%) |
| Client 4 | **83.5%** (2: 67%, 3: 100%) | **73.0%** (12: 94%, 13: 52%) | **80.5%** (24: 90%, 25: 71%) | **87.0%** (36: 90%, 37: 84%) | **59.0%** (40: 98%, 41: 20%) |
