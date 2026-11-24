# Server sync 9 — global model v9

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 87.97% on 36 observed labels.
Validation accuracy: 0.8777777777777778; seen labels never predicted: 0.
Server-accepted classes: 36; pending private classes: 0.
Accepted-only TEST accuracy: 0.8797222222222222.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.0%** (0: 89%, 1: 87%) | **90.0%** (4: 94%, 5: 86%) | **88.5%** (8: 96%, 9: 81%) | **79.0%** (12: 79%, 13: 79%) | **93.5%** (16: 89%, 17: 98%) | **94.5%** (20: 94%, 21: 95%) | **92.5%** (24: 90%, 25: 95%) | **87.5%** (28: 88%, 29: 87%) | **93.0%** (32: 92%, 33: 94%) |
| Client 1 | **91.5%** (2: 93%, 3: 90%) | **79.5%** (6: 78%, 7: 81%) | **78.5%** (10: 71%, 11: 86%) | **85.5%** (14: 87%, 15: 84%) | **84.0%** (18: 93%, 19: 75%) | **92.0%** (22: 89%, 23: 95%) | **85.5%** (26: 75%, 27: 96%) | **95.5%** (30: 99%, 31: 92%) | **85.0%** (34: 73%, 35: 97%) |
