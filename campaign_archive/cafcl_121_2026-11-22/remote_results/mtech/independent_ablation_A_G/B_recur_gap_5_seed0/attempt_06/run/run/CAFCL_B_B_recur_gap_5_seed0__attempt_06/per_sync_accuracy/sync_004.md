# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.20% on 40 observed labels.
Validation accuracy: 0.8155; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.812.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **88.0%** (0: 91%, 1: 85%) | **69.5%** (10: 89%, 11: 50%) | **93.5%** (20: 88%, 21: 99%) | **92.0%** (30: 97%, 31: 87%) |
| Client 1 | **86.0%** (2: 74%, 3: 98%) | **69.5%** (12: 92%, 13: 47%) | **88.0%** (22: 81%, 23: 95%) | **81.0%** (32: 70%, 33: 92%) |
| Client 2 | **83.5%** (4: 98%, 5: 69%) | **76.0%** (14: 63%, 15: 89%) | **86.0%** (24: 87%, 25: 85%) | **66.0%** (34: 100%, 35: 32%) |
| Client 3 | **79.0%** (6: 73%, 7: 85%) | **92.0%** (16: 87%, 17: 97%) | **75.5%** (26: 52%, 27: 99%) | **74.5%** (36: 56%, 37: 93%) |
| Client 4 | **81.5%** (8: 86%, 9: 77%) | **78.5%** (18: 88%, 19: 69%) | **80.0%** (28: 75%, 29: 85%) | **84.0%** (38: 86%, 39: 82%) |
