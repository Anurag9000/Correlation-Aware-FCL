# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.00% on 14 observed labels.
Validation accuracy: 0.5885714285714285; seen labels never predicted: 0.
Server-accepted classes: 14; pending private classes: 0.
Accepted-only TEST accuracy: 0.61.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **84.0%** (0: 75%, 1: 93%) | **56.0%** (10: 72%, 11: 40%) |
| Client 1 | **65.0%** (2: 30%, 3: 100%) | **67.0%** (4: 97%, 5: 37%) |
| Client 2 | **67.0%** (4: 97%, 5: 37%) | **52.5%** (8: 99%, 9: 6%) |
| Client 3 | **53.5%** (6: 8%, 7: 99%) | **49.0%** (12: 97%, 13: 1%) |
| Client 4 | **52.5%** (8: 99%, 9: 6%) | **65.0%** (2: 30%, 3: 100%) |
