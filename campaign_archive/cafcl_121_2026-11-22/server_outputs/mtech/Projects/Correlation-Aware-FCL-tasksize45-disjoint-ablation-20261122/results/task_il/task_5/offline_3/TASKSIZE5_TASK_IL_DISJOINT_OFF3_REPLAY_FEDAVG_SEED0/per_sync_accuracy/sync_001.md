# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.36% on 25 observed labels.
Validation accuracy: 0.2264; seen labels never predicted: 11.
Server-accepted classes: 10; pending private classes: 15.
Accepted-only TEST accuracy: 0.263.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **20.0%** (0: 0%, 1: 100%, 2: 0%, 3: 0%, 4: 0%) [PENDING upload] |
| Client 1 | **24.0%** (5: 5%, 6: 13%, 7: 2%, 8: 100%, 9: 0%) |
| Client 2 | **24.2%** (10: 2%, 11: 84%, 12: 1%, 13: 0%, 14: 34%) [PENDING upload] |
| Client 3 | **20.0%** (15: 0%, 16: 0%, 17: 100%, 18: 0%, 19: 0%) [PENDING upload] |
| Client 4 | **28.6%** (20: 0%, 21: 28%, 22: 17%, 23: 98%, 24: 0%) |
