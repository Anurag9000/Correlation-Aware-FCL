# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 31.00% on 40 observed labels.
Validation accuracy: 0.311; seen labels never predicted: 12.
Server-accepted classes: 20; pending private classes: 20.
Accepted-only TEST accuracy: 0.3615.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **26.2%** (0: 97%, 1: 8%, 2: 0%, 3: 0%) [PENDING upload] | **25.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] |
| Client 1 | **39.0%** (4: 99%, 5: 56%, 6: 1%, 7: 0%) | **25.5%** (24: 2%, 25: 0%, 26: 100%, 27: 0%) [PENDING upload] |
| Client 2 | **34.0%** (8: 86%, 9: 2%, 10: 4%, 11: 44%) | **30.2%** (28: 8%, 29: 42%, 30: 0%, 31: 71%) |
| Client 3 | **25.0%** (12: 0%, 13: 0%, 14: 100%, 15: 0%) [PENDING upload] | **27.5%** (32: 46%, 33: 2%, 34: 62%, 35: 0%) [PENDING upload] |
| Client 4 | **36.8%** (16: 0%, 17: 2%, 18: 76%, 19: 69%) | **40.8%** (36: 0%, 37: 14%, 38: 90%, 39: 59%) |
