# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.08% on 12 observed labels.
Validation accuracy: 0.3883333333333333; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.43083333333333335.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **61.0%** (0: 68%, 1: 54%) | **59.5%** (4: 46%, 5: 73%) | **11.5%** (8: 7%, 9: 16%) |
| Client 1 | **57.5%** (2: 48%, 3: 67%) | **56.5%** (6: 55%, 7: 58%) | **12.5%** (10: 11%, 11: 14%) |
