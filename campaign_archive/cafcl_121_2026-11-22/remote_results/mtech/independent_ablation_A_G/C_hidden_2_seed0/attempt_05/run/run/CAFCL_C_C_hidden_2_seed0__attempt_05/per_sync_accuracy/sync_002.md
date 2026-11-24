# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.35% on 20 observed labels.
Validation accuracy: 0.598; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.63.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.5%** (8: 100%, 9: 3%) | **69.5%** (14: 56%, 15: 83%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **49.5%** (18: 86%, 19: 13%) [PENDING upload] |
| Client 2 | **62.0%** (4: 98%, 5: 26%) | **67.0%** (16: 34%, 17: 100%) |
| Client 3 | **79.0%** (0: 69%, 1: 89%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **72.0%** (2: 44%, 3: 100%) | **53.0%** (12: 94%, 13: 12%) |
