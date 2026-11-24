# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.65% on 20 observed labels.
Validation accuracy: 0.515; seen labels never predicted: 9.
Server-accepted classes: 18; pending private classes: 2.
Accepted-only TEST accuracy: 0.5183333333333333.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **50.0%** (12: 100%, 13: 0%) |
| Client 2 | **66.5%** (4: 94%, 5: 39%) | **50.0%** (14: 100%, 15: 0%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **50.0%** (16: 100%, 17: 0%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) | **50.0%** (18: 0%, 19: 100%) |
