# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.73% on 30 observed labels.
Validation accuracy: 0.8106666666666666; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.8073333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 82%, 1: 86%) | **84.0%** (0: 82%, 1: 86%) | **68.0%** (10: 91%, 11: 45%) | **68.0%** (10: 91%, 11: 45%) | **93.5%** (20: 91%, 21: 96%) |
| Client 1 | **88.0%** (2: 81%, 3: 95%) | **88.0%** (2: 81%, 3: 95%) | **71.5%** (12: 98%, 13: 45%) | **71.5%** (12: 98%, 13: 45%) | **84.0%** (22: 75%, 23: 93%) |
| Client 2 | **86.5%** (4: 89%, 5: 84%) | **86.5%** (4: 89%, 5: 84%) | **76.5%** (14: 58%, 15: 95%) | **76.5%** (14: 58%, 15: 95%) | **81.0%** (24: 65%, 25: 97%) |
| Client 3 | **80.0%** (6: 74%, 7: 86%) | **80.0%** (6: 74%, 7: 86%) | **90.5%** (16: 86%, 17: 95%) | **90.5%** (16: 86%, 17: 95%) | **73.0%** (26: 87%, 27: 59%) |
| Client 4 | **82.0%** (8: 78%, 9: 86%) | **82.0%** (8: 78%, 9: 86%) | **78.0%** (18: 84%, 19: 72%) | **78.0%** (18: 84%, 19: 72%) | **74.5%** (28: 57%, 29: 92%) |
