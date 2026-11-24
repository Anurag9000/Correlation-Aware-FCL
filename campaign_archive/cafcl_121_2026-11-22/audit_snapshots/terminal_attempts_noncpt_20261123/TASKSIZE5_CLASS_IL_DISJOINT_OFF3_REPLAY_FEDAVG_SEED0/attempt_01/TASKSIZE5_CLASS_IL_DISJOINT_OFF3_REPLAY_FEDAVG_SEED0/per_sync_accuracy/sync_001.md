# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 4.68% on 25 observed labels.
Validation accuracy: 0.0456; seen labels never predicted: 20.
Server-accepted classes: 10; pending private classes: 15.
Accepted-only TEST accuracy: 0.117.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%, 2: 0%, 3: 0%, 4: 0%) [PENDING upload] |
| Client 1 | **2.0%** (5: 6%, 6: 4%, 7: 0%, 8: 0%, 9: 0%) |
| Client 2 | **0.0%** (10: 0%, 11: 0%, 12: 0%, 13: 0%, 14: 0%) [PENDING upload] |
| Client 3 | **0.0%** (15: 0%, 16: 0%, 17: 0%, 18: 0%, 19: 0%) [PENDING upload] |
| Client 4 | **21.4%** (20: 0%, 21: 0%, 22: 7%, 23: 100%, 24: 0%) |
