# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 64.70% on 10 observed labels.
Validation accuracy: 0.652; seen labels never predicted: 1.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.647.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **75.5%** (0: 88%, 1: 63%) |
| Client 1 | **73.0%** (2: 92%, 3: 54%) |
| Client 2 | **72.0%** (4: 91%, 5: 53%) |
| Client 3 | **50.0%** (6: 100%, 7: 0%) |
| Client 4 | **53.0%** (8: 99%, 9: 7%) |
