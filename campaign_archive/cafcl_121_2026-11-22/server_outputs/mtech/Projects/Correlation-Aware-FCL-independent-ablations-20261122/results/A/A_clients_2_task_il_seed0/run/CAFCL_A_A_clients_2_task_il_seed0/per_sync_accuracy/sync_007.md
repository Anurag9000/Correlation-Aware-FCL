# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.75% on 28 observed labels.
Validation accuracy: 0.8592857142857143; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.8575.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.5%** (0: 82%, 1: 87%) | **92.0%** (4: 95%, 5: 89%) | **85.5%** (8: 90%, 9: 81%) | **81.0%** (12: 86%, 13: 76%) | **94.0%** (16: 90%, 17: 98%) | **96.0%** (20: 97%, 21: 95%) | **88.0%** (24: 93%, 25: 83%) |
| Client 1 | **89.5%** (2: 83%, 3: 96%) | **77.5%** (6: 80%, 7: 75%) | **74.0%** (10: 84%, 11: 64%) | **86.0%** (14: 80%, 15: 92%) | **85.5%** (18: 95%, 19: 76%) | **89.5%** (22: 85%, 23: 94%) | **77.5%** (26: 68%, 27: 87%) |
