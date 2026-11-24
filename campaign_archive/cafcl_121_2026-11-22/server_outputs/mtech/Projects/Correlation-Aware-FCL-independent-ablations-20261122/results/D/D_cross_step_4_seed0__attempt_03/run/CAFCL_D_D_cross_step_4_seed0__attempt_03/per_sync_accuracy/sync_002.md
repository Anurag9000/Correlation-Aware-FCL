# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 65.75% on 12 observed labels.
Validation accuracy: 0.645; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.6575.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **82.0%** (0: 69%, 1: 95%) | **72.0%** (2: 44%, 3: 100%) |
| Client 1 | **72.0%** (2: 44%, 3: 100%) | **58.5%** (4: 38%, 5: 79%) |
| Client 2 | **58.5%** (4: 38%, 5: 79%) | **61.5%** (8: 93%, 9: 30%) |
| Client 3 | **60.0%** (6: 24%, 7: 96%) | **60.5%** (10: 65%, 11: 56%) |
| Client 4 | **61.5%** (8: 93%, 9: 30%) | **82.0%** (0: 69%, 1: 95%) |
