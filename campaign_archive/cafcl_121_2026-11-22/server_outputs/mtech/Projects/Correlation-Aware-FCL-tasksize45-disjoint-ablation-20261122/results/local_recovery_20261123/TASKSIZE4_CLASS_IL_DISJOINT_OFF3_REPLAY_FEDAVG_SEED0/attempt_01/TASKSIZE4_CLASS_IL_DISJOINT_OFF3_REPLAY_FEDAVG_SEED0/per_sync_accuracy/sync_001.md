# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 9.25% on 20 observed labels.
Validation accuracy: 0.078; seen labels never predicted: 16.
Server-accepted classes: 8; pending private classes: 12.
Accepted-only TEST accuracy: 0.23125.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%) [PENDING upload] |
| Client 1 | **24.5%** (4: 30%, 5: 68%, 6: 0%, 7: 0%) |
| Client 2 | **0.0%** (8: 0%, 9: 0%, 10: 0%, 11: 0%) [PENDING upload] |
| Client 3 | **0.0%** (12: 0%, 13: 0%, 14: 0%, 15: 0%) [PENDING upload] |
| Client 4 | **21.8%** (16: 0%, 17: 0%, 18: 87%, 19: 0%) |
