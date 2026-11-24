# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.56% on 25 observed labels.
Validation accuracy: 0.2064; seen labels never predicted: 15.
Server-accepted classes: 15; pending private classes: 10.
Accepted-only TEST accuracy: 0.23.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **20.4%** (0: 0%, 1: 0%, 2: 2%, 3: 0%, 4: 100%) |
| Client 1 | **26.4%** (5: 34%, 6: 2%, 7: 0%, 8: 96%, 9: 0%) |
| Client 2 | **20.0%** (10: 0%, 11: 100%, 12: 0%, 13: 0%, 14: 0%) [PENDING upload] |
| Client 3 | **18.8%** (15: 0%, 16: 86%, 17: 8%, 18: 0%, 19: 0%) [PENDING upload] |
| Client 4 | **22.2%** (20: 0%, 21: 0%, 22: 15%, 23: 96%, 24: 0%) |
