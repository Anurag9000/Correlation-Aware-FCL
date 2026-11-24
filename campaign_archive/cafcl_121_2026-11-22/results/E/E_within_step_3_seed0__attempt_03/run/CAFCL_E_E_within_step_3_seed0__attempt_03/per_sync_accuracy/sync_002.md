# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.92% on 12 observed labels.
Validation accuracy: 0.6566666666666666; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.6691666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **75.0%** (0: 50%, 1: 100%) | **57.0%** (6: 34%, 7: 80%) |
| Client 1 | **75.0%** (0: 50%, 1: 100%) | **57.0%** (6: 34%, 7: 80%) |
| Client 2 | **72.5%** (4: 92%, 5: 53%) | **57.0%** (6: 34%, 7: 80%) |
| Client 3 | **75.0%** (0: 50%, 1: 100%) | **64.0%** (10: 62%, 11: 66%) |
| Client 4 | **78.0%** (2: 56%, 3: 100%) | **55.0%** (8: 98%, 9: 12%) |
