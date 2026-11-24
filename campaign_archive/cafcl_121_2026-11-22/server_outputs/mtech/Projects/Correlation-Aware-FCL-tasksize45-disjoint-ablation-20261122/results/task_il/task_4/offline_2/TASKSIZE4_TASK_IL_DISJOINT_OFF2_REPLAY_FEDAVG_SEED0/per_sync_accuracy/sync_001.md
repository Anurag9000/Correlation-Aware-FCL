# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.85% on 20 observed labels.
Validation accuracy: 0.296; seen labels never predicted: 8.
Server-accepted classes: 12; pending private classes: 8.
Accepted-only TEST accuracy: 0.3308333333333333.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **33.2%** (0: 0%, 1: 0%, 2: 35%, 3: 98%) |
| Client 1 | **33.5%** (4: 56%, 5: 1%, 6: 2%, 7: 75%) |
| Client 2 | **25.5%** (8: 0%, 9: 0%, 10: 94%, 11: 8%) [PENDING upload] |
| Client 3 | **29.5%** (12: 0%, 13: 0%, 14: 40%, 15: 78%) [PENDING upload] |
| Client 4 | **32.5%** (16: 33%, 17: 0%, 18: 97%, 19: 0%) |
