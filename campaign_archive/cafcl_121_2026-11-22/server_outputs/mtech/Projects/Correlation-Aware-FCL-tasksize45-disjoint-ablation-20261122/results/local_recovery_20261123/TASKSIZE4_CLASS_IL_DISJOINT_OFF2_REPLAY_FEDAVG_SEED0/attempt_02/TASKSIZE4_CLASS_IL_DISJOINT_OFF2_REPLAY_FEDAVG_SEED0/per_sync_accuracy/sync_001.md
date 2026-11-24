# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.05% on 20 observed labels.
Validation accuracy: 0.055; seen labels never predicted: 15.
Server-accepted classes: 12; pending private classes: 8.
Accepted-only TEST accuracy: 0.10083333333333333.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **10.2%** (0: 0%, 1: 0%, 2: 0%, 3: 41%) |
| Client 1 | **19.5%** (4: 78%, 5: 0%, 6: 0%, 7: 0%) |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] |
| Client 3 | **0.0%** (12: 0%, 13: 0%, 14: 0%, 15: 0%) [PENDING upload] |
| Client 4 | **0.5%** (16: 0%, 17: 0%, 18: 2%, 19: 0%) |
