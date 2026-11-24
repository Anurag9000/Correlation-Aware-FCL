# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.17% on 30 observed labels.
Validation accuracy: 0.7833333333333333; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.7816666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 72%, 1: 96%) | **62.5%** (10: 90%, 11: 35%) | **89.0%** (20: 79%, 21: 99%) |
| Client 1 | **77.5%** (2: 55%, 3: 100%) | **67.5%** (12: 93%, 13: 42%) | **87.0%** (22: 91%, 23: 83%) |
| Client 2 | **86.0%** (4: 94%, 5: 78%) | **74.5%** (14: 62%, 15: 87%) | **83.5%** (24: 88%, 25: 79%) |
| Client 3 | **72.5%** (6: 52%, 7: 93%) | **87.5%** (16: 79%, 17: 96%) | **75.0%** (26: 51%, 27: 99%) |
| Client 4 | **74.5%** (8: 96%, 9: 53%) | **76.0%** (18: 86%, 19: 66%) | **75.5%** (28: 58%, 29: 93%) |
