# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.88% on 25 observed labels.
Validation accuracy: 0.2376; seen labels never predicted: 12.
Server-accepted classes: 10; pending private classes: 15.
Accepted-only TEST accuracy: 0.302.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **24.2%** (0: 47%, 1: 74%, 2: 0%, 3: 0%, 4: 0%) [PENDING upload] |
| Client 1 | **38.8%** (5: 76%, 6: 35%, 7: 3%, 8: 80%, 9: 0%) |
| Client 2 | **19.8%** (10: 29%, 11: 69%, 12: 0%, 13: 1%, 14: 0%) [PENDING upload] |
| Client 3 | **20.0%** (15: 0%, 16: 0%, 17: 100%, 18: 0%, 19: 0%) [PENDING upload] |
| Client 4 | **21.6%** (20: 0%, 21: 0%, 22: 8%, 23: 100%, 24: 0%) |
