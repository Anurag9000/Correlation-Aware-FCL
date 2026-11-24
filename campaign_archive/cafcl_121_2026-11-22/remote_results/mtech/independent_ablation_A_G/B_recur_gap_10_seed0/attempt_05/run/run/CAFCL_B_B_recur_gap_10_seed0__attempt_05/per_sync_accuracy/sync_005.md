# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.42% on 50 observed labels.
Validation accuracy: 0.8536; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8542.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **91.0%** (0: 87%, 1: 95%) | **73.5%** (10: 92%, 11: 55%) | **95.5%** (20: 91%, 21: 100%) | **92.5%** (30: 98%, 31: 87%) | **88.5%** (40: 96%, 41: 81%) |
| Client 1 | **90.0%** (2: 85%, 3: 95%) | **79.5%** (12: 88%, 13: 71%) | **84.5%** (22: 71%, 23: 98%) | **87.5%** (32: 79%, 33: 96%) | **74.5%** (42: 85%, 43: 64%) |
| Client 2 | **88.5%** (4: 89%, 5: 88%) | **78.0%** (14: 59%, 15: 97%) | **85.0%** (24: 80%, 25: 90%) | **83.5%** (34: 90%, 35: 77%) | **71.5%** (44: 92%, 45: 51%) |
| Client 3 | **82.5%** (6: 85%, 7: 80%) | **93.0%** (16: 90%, 17: 96%) | **82.0%** (26: 72%, 27: 92%) | **90.5%** (36: 86%, 37: 95%) | **95.0%** (46: 97%, 47: 93%) |
| Client 4 | **85.0%** (8: 86%, 9: 84%) | **81.5%** (18: 77%, 19: 86%) | **83.5%** (28: 92%, 29: 75%) | **85.5%** (38: 81%, 39: 90%) | **93.5%** (48: 90%, 49: 97%) |
