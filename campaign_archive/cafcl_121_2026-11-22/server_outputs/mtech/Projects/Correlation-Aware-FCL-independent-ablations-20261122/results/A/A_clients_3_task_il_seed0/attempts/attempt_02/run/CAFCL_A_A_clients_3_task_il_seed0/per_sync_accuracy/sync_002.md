# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.83% on 12 observed labels.
Validation accuracy: 0.63; seen labels never predicted: 0.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.6183333333333333.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **63.0%** (0: 26%, 1: 100%) | **67.0%** (6: 93%, 7: 41%) |
| Client 1 | **58.5%** (2: 17%, 3: 100%) | **60.5%** (8: 99%, 9: 22%) |
| Client 2 | **67.5%** (4: 97%, 5: 38%) | **54.5%** (10: 95%, 11: 14%) |
