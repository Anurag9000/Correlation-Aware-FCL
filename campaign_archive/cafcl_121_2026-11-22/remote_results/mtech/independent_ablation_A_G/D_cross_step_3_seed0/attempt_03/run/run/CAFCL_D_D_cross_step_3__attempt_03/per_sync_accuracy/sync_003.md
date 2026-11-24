# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.39% on 18 observed labels.
Validation accuracy: 0.75; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.7438888888888889.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **75.0%** (0: 57%, 1: 93%) | **61.5%** (10: 75%, 11: 48%) | **77.0%** (14: 70%, 15: 84%) |
| Client 1 | **88.0%** (2: 84%, 3: 92%) | **82.5%** (4: 88%, 5: 77%) | **88.0%** (2: 84%, 3: 92%) |
| Client 2 | **82.5%** (4: 88%, 5: 77%) | **71.5%** (8: 97%, 9: 46%) | **59.0%** (12: 93%, 13: 25%) |
| Client 3 | **72.5%** (6: 60%, 7: 85%) | **59.0%** (12: 93%, 13: 25%) | **82.5%** (16: 73%, 17: 92%) |
| Client 4 | **71.5%** (8: 97%, 9: 46%) | **88.0%** (2: 84%, 3: 92%) | **61.5%** (10: 75%, 11: 48%) |
