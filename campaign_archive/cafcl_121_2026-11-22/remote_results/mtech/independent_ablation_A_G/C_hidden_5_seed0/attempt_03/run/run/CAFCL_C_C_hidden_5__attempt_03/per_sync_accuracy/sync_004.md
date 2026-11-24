# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.22% on 40 observed labels.
Validation accuracy: 0.7655; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 8.
Accepted-only TEST accuracy: 0.8290625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **84.5%** (8: 94%, 9: 75%) | **75.0%** (14: 71%, 15: 79%) | **77.0%** (26: 60%, 27: 94%) | **81.5%** (34: 86%, 35: 77%) |
| Client 1 | **48.5%** (6: 0%, 7: 97%) [PENDING upload] | **53.5%** (18: 11%, 19: 96%) [PENDING upload] | **45.5%** (28: 47%, 29: 44%) [PENDING upload] | **50.5%** (30: 100%, 31: 1%) [PENDING upload] |
| Client 2 | **81.0%** (4: 95%, 5: 67%) | **92.5%** (16: 87%, 17: 98%) | **87.5%** (22: 84%, 23: 91%) | **76.0%** (32: 56%, 33: 96%) |
| Client 3 | **87.5%** (0: 92%, 1: 83%) | **80.0%** (10: 85%, 11: 75%) | **93.0%** (20: 88%, 21: 98%) | **82.0%** (38: 88%, 39: 76%) |
| Client 4 | **82.0%** (2: 73%, 3: 91%) | **70.0%** (12: 78%, 13: 62%) | **89.5%** (24: 90%, 25: 89%) | **87.5%** (36: 81%, 37: 94%) |
