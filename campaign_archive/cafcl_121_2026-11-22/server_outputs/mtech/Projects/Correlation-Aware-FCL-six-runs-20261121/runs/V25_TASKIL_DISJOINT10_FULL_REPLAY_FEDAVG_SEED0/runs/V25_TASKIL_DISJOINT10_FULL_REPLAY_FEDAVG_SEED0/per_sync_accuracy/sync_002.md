# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.55% on 20 observed labels.
Validation accuracy: 0.561; seen labels never predicted: 2.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5655.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.5%** (0: 17%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **55.5%** (2: 11%, 3: 100%) | **50.0%** (12: 98%, 13: 2%) |
| Client 2 | **65.0%** (4: 100%, 5: 30%) | **62.5%** (14: 27%, 15: 98%) |
| Client 3 | **64.5%** (6: 32%, 7: 97%) | **56.0%** (16: 94%, 17: 18%) |
| Client 4 | **53.5%** (8: 98%, 9: 9%) | **50.0%** (18: 100%, 19: 0%) |
