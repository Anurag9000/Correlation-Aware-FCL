# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.31% on 16 observed labels.
Validation accuracy: 0.53875; seen labels never predicted: 4.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.5585714285714286.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **92.5%** (0: 86%, 1: 99%) [PENDING upload] | **57.0%** (5: 20%, 6: 94%) | **49.5%** (10: 99%, 11: 0%) |
| Client 1 | **82.5%** (1: 99%, 2: 66%) | **94.5%** (6: 94%, 7: 95%) | **50.0%** (11: 0%, 12: 100%) |
| Client 2 | **33.0%** (2: 66%, 3: 0%) | **47.5%** (7: 95%, 8: 0%) | **70.5%** (12: 100%, 13: 41%) |
| Client 3 | **50.0%** (3: 0%, 4: 100%) | **2.0%** (8: 0%, 9: 4%) | **69.0%** (13: 41%, 14: 97%) |
| Client 4 | **60.0%** (4: 100%, 5: 20%) | **51.5%** (9: 4%, 10: 99%) | **48.5%** (14: 97%, 15: 0%) [PENDING upload] |
