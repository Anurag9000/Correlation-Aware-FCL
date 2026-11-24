# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.92% on 12 observed labels.
Validation accuracy: 0.7616666666666667; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.7391666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **86.0%** (0: 80%, 1: 92%) | **84.0%** (4: 93%, 5: 75%) | **59.0%** (8: 96%, 9: 22%) |
| Client 1 | **86.0%** (0: 80%, 1: 92%) | **84.0%** (4: 93%, 5: 75%) | **59.0%** (8: 96%, 9: 22%) |
| Client 2 | **80.5%** (2: 63%, 3: 98%) | **84.0%** (4: 93%, 5: 75%) | **65.5%** (10: 62%, 11: 69%) |
| Client 3 | **86.0%** (0: 80%, 1: 92%) | **68.5%** (6: 43%, 7: 94%) | **59.0%** (8: 96%, 9: 22%) |
| Client 4 | **86.0%** (0: 80%, 1: 92%) | **84.0%** (4: 93%, 5: 75%) | **59.0%** (8: 96%, 9: 22%) |
