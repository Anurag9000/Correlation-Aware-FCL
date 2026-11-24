# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 49.90% on 20 observed labels.
Validation accuracy: 0.501; seen labels never predicted: 7.
Server-accepted classes: 14; pending private classes: 6.
Accepted-only TEST accuracy: 0.5014285714285714.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) [PENDING upload] |
| Client 1 | **50.5%** (2: 1%, 3: 100%) | **50.0%** (12: 0%, 13: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) [PENDING upload] | **48.0%** (14: 4%, 15: 92%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **50.0%** (16: 100%, 17: 0%) |
| Client 4 | **50.5%** (8: 100%, 9: 1%) | **50.0%** (18: 100%, 19: 0%) |
