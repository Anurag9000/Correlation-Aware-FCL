# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.79% on 28 observed labels.
Validation accuracy: 0.8114285714285714; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.8078571428571428.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 83%, 1: 85%) | **77.5%** (8: 97%, 9: 58%) | **94.5%** (16: 92%, 17: 97%) | **76.5%** (22: 60%, 23: 93%) |
| Client 1 | **87.0%** (2: 78%, 3: 96%) | **70.0%** (10: 73%, 11: 67%) | **77.5%** (8: 97%, 9: 58%) | **94.5%** (16: 92%, 17: 97%) |
| Client 2 | **85.0%** (4: 87%, 5: 83%) | **69.0%** (12: 95%, 13: 43%) | **82.0%** (18: 85%, 19: 79%) | **93.0%** (20: 87%, 21: 99%) |
| Client 3 | **75.0%** (6: 58%, 7: 92%) | **79.5%** (14: 77%, 15: 82%) | **93.0%** (20: 87%, 21: 99%) | **87.0%** (24: 90%, 25: 84%) |
| Client 4 | **77.5%** (8: 97%, 9: 58%) | **85.0%** (4: 87%, 5: 83%) | **69.0%** (12: 95%, 13: 43%) | **71.0%** (26: 43%, 27: 99%) |
