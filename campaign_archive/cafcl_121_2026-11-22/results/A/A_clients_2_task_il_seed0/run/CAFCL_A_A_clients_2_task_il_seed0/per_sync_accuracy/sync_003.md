# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.17% on 12 observed labels.
Validation accuracy: 0.7883333333333333; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.7716666666666666.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **65.5%** (0: 34%, 1: 97%) | **85.5%** (4: 94%, 5: 77%) | **79.5%** (8: 93%, 9: 66%) |
| Client 1 | **87.0%** (2: 76%, 3: 98%) | **76.0%** (6: 69%, 7: 83%) | **69.5%** (10: 78%, 11: 61%) |
