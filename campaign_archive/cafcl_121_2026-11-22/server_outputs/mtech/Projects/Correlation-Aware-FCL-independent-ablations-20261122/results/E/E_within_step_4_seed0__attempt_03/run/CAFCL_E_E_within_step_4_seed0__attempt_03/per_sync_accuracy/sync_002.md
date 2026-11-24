# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.12% on 8 observed labels.
Validation accuracy: 0.7175; seen labels never predicted: 0.
Server-accepted classes: 8; pending private classes: 0.
Accepted-only TEST accuracy: 0.74125.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **84.0%** (0: 73%, 1: 95%) | **71.5%** (4: 95%, 5: 48%) |
| Client 1 | **84.0%** (0: 73%, 1: 95%) | **71.5%** (4: 95%, 5: 48%) |
| Client 2 | **88.0%** (2: 81%, 3: 95%) | **71.5%** (4: 95%, 5: 48%) |
| Client 3 | **84.0%** (0: 73%, 1: 95%) | **53.0%** (6: 6%, 7: 100%) |
| Client 4 | **84.0%** (0: 73%, 1: 95%) | **71.5%** (4: 95%, 5: 48%) |
