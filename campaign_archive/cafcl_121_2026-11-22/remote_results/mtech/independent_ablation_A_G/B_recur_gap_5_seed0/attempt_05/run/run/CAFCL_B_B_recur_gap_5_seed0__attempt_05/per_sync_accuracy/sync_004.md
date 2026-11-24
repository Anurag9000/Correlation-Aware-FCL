# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.38% on 40 observed labels.
Validation accuracy: 0.8025; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.80375.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 88%, 1: 90%) | **69.0%** (10: 86%, 11: 52%) | **94.0%** (20: 88%, 21: 100%) | **91.5%** (30: 93%, 31: 90%) |
| Client 1 | **83.0%** (2: 68%, 3: 98%) | **69.5%** (12: 95%, 13: 44%) | **86.0%** (22: 76%, 23: 96%) | **76.5%** (32: 61%, 33: 92%) |
| Client 2 | **80.5%** (4: 97%, 5: 64%) | **71.0%** (14: 49%, 15: 93%) | **85.5%** (24: 84%, 25: 87%) | **64.0%** (34: 99%, 35: 29%) |
| Client 3 | **80.0%** (6: 78%, 7: 82%) | **89.0%** (16: 81%, 17: 97%) | **74.5%** (26: 50%, 27: 99%) | **82.0%** (36: 72%, 37: 92%) |
| Client 4 | **82.5%** (8: 91%, 9: 74%) | **77.5%** (18: 79%, 19: 76%) | **79.5%** (28: 75%, 29: 84%) | **83.0%** (38: 85%, 39: 81%) |
