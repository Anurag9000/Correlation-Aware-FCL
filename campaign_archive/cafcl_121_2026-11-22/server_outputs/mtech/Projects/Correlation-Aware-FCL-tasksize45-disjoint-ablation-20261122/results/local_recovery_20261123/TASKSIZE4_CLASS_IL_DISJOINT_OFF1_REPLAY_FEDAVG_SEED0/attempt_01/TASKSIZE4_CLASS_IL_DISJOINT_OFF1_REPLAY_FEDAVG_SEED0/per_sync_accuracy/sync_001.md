# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 7.05% on 20 observed labels.
Validation accuracy: 0.066; seen labels never predicted: 18.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.088125.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **22.5%** (0: 0%, 1: 0%, 2: 0%, 3: 90%) |
| Client 1 | **0.0%** (4: 0%, 5: 0%, 6: 0%, 7: 0%) |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) |
| Client 3 | **12.8%** (12: 0%, 13: 0%, 14: 51%, 15: 0%) |
| Client 4 | **0.0%** (16: 0%, 17: 0%, 18: 0%, 19: 0%) [PENDING upload] |
