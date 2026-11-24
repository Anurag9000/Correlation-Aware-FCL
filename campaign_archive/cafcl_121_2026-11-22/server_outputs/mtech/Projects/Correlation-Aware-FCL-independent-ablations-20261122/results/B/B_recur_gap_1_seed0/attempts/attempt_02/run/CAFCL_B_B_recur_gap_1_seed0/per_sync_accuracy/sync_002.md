# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.50% on 20 observed labels.
Validation accuracy: 0.591; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.595.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **61.0%** (0: 22%, 1: 100%) | **50.0%** (10: 99%, 11: 1%) |
| Client 1 | **54.5%** (2: 9%, 3: 100%) | **50.5%** (12: 95%, 13: 6%) |
| Client 2 | **70.5%** (4: 92%, 5: 49%) | **57.5%** (14: 15%, 15: 100%) |
| Client 3 | **53.5%** (6: 10%, 7: 97%) | **81.0%** (16: 66%, 17: 96%) |
| Client 4 | **52.5%** (8: 100%, 9: 5%) | **64.0%** (18: 50%, 19: 78%) |
