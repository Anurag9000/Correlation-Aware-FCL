# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.70% on 40 observed labels.
Validation accuracy: 0.8275; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.827.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **88.0%** (0: 91%, 1: 85%) | **72.5%** (10: 79%, 11: 66%) | **95.0%** (20: 91%, 21: 99%) | **92.0%** (30: 93%, 31: 91%) |
| Client 1 | **92.0%** (2: 86%, 3: 98%) | **70.0%** (12: 90%, 13: 50%) | **87.5%** (22: 81%, 23: 94%) | **79.5%** (32: 72%, 33: 87%) |
| Client 2 | **87.5%** (4: 94%, 5: 81%) | **76.0%** (14: 58%, 15: 94%) | **85.0%** (24: 81%, 25: 89%) | **78.0%** (34: 93%, 35: 63%) |
| Client 3 | **79.0%** (6: 80%, 7: 78%) | **93.0%** (16: 88%, 17: 98%) | **75.0%** (26: 51%, 27: 99%) | **83.0%** (36: 74%, 37: 92%) |
| Client 4 | **83.5%** (8: 92%, 9: 75%) | **76.5%** (18: 79%, 19: 74%) | **78.5%** (28: 74%, 29: 83%) | **82.5%** (38: 82%, 39: 83%) |
