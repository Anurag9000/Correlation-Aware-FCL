# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.95% on 20 observed labels.
Validation accuracy: 0.511; seen labels never predicted: 5.
Server-accepted classes: 18; pending private classes: 2.
Accepted-only TEST accuracy: 0.5177777777777778.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **48.5%** (12: 95%, 13: 2%) |
| Client 2 | **58.0%** (4: 95%, 5: 21%) | **43.5%** (14: 76%, 15: 11%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **59.0%** (16: 19%, 17: 99%) |
| Client 4 | **50.5%** (8: 100%, 9: 1%) | **50.0%** (18: 0%, 19: 100%) |
