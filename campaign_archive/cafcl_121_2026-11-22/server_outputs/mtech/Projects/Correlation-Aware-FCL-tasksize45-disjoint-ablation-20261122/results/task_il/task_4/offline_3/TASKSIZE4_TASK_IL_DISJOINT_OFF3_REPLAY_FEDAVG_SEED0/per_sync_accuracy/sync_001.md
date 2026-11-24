# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 32.60% on 20 observed labels.
Validation accuracy: 0.307; seen labels never predicted: 7.
Server-accepted classes: 8; pending private classes: 12.
Accepted-only TEST accuracy: 0.4475.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **23.8%** (0: 94%, 1: 0%, 2: 1%, 3: 0%) [PENDING upload] |
| Client 1 | **47.8%** (4: 90%, 5: 45%, 6: 2%, 7: 54%) |
| Client 2 | **24.8%** (8: 0%, 9: 0%, 10: 4%, 11: 95%) [PENDING upload] |
| Client 3 | **25.0%** (12: 0%, 13: 0%, 14: 100%, 15: 0%) [PENDING upload] |
| Client 4 | **41.8%** (16: 87%, 17: 2%, 18: 78%, 19: 0%) |
