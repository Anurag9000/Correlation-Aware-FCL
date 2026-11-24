# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.60% on 25 observed labels.
Validation accuracy: 0.2368; seen labels never predicted: 13.
Server-accepted classes: 20; pending private classes: 5.
Accepted-only TEST accuracy: 0.245.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **27.8%** (0: 0%, 1: 0%, 2: 45%, 3: 3%, 4: 91%) |
| Client 1 | **23.8%** (5: 0%, 6: 58%, 7: 55%, 8: 6%, 9: 0%) |
| Client 2 | **20.0%** (10: 1%, 11: 0%, 12: 99%, 13: 0%, 14: 0%) |
| Client 3 | **26.4%** (15: 55%, 16: 0%, 17: 0%, 18: 77%, 19: 0%) |
| Client 4 | **20.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%, 24: 0%) [PENDING upload] |
