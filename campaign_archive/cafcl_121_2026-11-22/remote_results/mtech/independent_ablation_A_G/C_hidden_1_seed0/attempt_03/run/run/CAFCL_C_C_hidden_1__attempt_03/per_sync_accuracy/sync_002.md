# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.05% on 20 observed labels.
Validation accuracy: 0.551; seen labels never predicted: 6.
Server-accepted classes: 18; pending private classes: 2.
Accepted-only TEST accuracy: 0.5377777777777778.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (8: 100%, 9: 0%) | **66.5%** (14: 75%, 15: 58%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) | **50.0%** (18: 0%, 19: 100%) |
| Client 2 | **67.0%** (4: 94%, 5: 40%) | **50.5%** (16: 1%, 17: 100%) |
| Client 3 | **50.0%** (0: 0%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **50.0%** (2: 0%, 3: 100%) | **46.5%** (12: 74%, 13: 19%) [PENDING upload] |
