# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 86.88% on 42 observed labels.
Validation accuracy: 0.878095238095238; seen labels never predicted: 0.
Server-accepted classes: 42; pending private classes: 0.
Accepted-only TEST accuracy: 0.8688095238095238.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.5%** (0: 77%, 1: 96%) | **83.0%** (6: 87%, 7: 79%) | **78.5%** (12: 95%, 13: 62%) | **82.5%** (18: 74%, 19: 91%) | **88.5%** (24: 92%, 25: 85%) | **92.5%** (30: 93%, 31: 92%) | **89.0%** (36: 85%, 37: 93%) |
| Client 1 | **90.5%** (2: 84%, 3: 97%) | **91.0%** (8: 97%, 9: 85%) | **85.0%** (14: 76%, 15: 94%) | **96.0%** (20: 92%, 21: 100%) | **83.5%** (26: 78%, 27: 89%) | **91.0%** (32: 89%, 33: 93%) | **82.5%** (38: 88%, 39: 77%) |
| Client 2 | **88.0%** (4: 91%, 5: 85%) | **79.5%** (10: 80%, 11: 79%) | **94.0%** (16: 90%, 17: 98%) | **87.5%** (22: 76%, 23: 99%) | **83.5%** (28: 85%, 29: 82%) | **81.5%** (34: 83%, 35: 80%) | **90.5%** (40: 95%, 41: 86%) |
