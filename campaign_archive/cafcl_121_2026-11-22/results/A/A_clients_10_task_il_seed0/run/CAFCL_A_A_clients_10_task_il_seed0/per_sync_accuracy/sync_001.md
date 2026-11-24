# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 50.55% on 20 observed labels.
Validation accuracy: 0.509; seen labels never predicted: 8.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5055.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) |
| Client 4 | **50.0%** (8: 0%, 9: 100%) |
| Client 5 | **50.0%** (10: 0%, 11: 100%) |
| Client 6 | **50.0%** (12: 100%, 13: 0%) |
| Client 7 | **50.0%** (14: 100%, 15: 0%) |
| Client 8 | **55.0%** (16: 94%, 17: 16%) |
| Client 9 | **50.5%** (18: 99%, 19: 2%) |
