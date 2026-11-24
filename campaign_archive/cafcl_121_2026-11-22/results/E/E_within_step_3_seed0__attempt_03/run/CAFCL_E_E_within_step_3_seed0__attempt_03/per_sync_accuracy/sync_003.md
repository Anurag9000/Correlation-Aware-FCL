# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.00% on 18 observed labels.
Validation accuracy: 0.8033333333333333; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.81.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **86.0%** (0: 79%, 1: 93%) | **78.0%** (6: 80%, 7: 76%) | **63.5%** (12: 92%, 13: 35%) |
| Client 1 | **86.0%** (0: 79%, 1: 93%) | **78.0%** (6: 80%, 7: 76%) | **63.5%** (12: 92%, 13: 35%) |
| Client 2 | **84.0%** (4: 95%, 5: 73%) | **78.0%** (6: 80%, 7: 76%) | **87.5%** (16: 81%, 17: 94%) |
| Client 3 | **86.0%** (0: 79%, 1: 93%) | **77.0%** (10: 87%, 11: 67%) | **80.0%** (14: 68%, 15: 92%) |
| Client 4 | **88.5%** (2: 83%, 3: 94%) | **84.5%** (8: 96%, 9: 73%) | **63.5%** (12: 92%, 13: 35%) |
