# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.56% on 16 observed labels.
Validation accuracy: 0.8125; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.805625.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 81%, 1: 86%) | **89.0%** (4: 94%, 5: 84%) | **81.5%** (8: 91%, 9: 72%) | **76.0%** (12: 92%, 13: 60%) |
| Client 1 | **86.0%** (2: 90%, 3: 82%) | **78.5%** (6: 76%, 7: 81%) | **69.5%** (10: 78%, 11: 61%) | **80.5%** (14: 91%, 15: 70%) |
