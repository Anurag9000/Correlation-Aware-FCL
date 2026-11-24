# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 32.42% on 40 observed labels.
Validation accuracy: 0.316; seen labels never predicted: 12.
Server-accepted classes: 36; pending private classes: 4.
Accepted-only TEST accuracy: 0.3313888888888889.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **30.8%** (0: 1%, 1: 0%, 2: 22%, 3: 100%) | **42.5%** (20: 0%, 21: 92%, 22: 78%, 23: 0%) |
| Client 1 | **28.2%** (4: 100%, 5: 11%, 6: 0%, 7: 2%) | **31.2%** (24: 0%, 25: 10%, 26: 16%, 27: 99%) |
| Client 2 | **27.0%** (8: 99%, 9: 3%, 10: 0%, 11: 6%) | **26.0%** (28: 97%, 29: 7%, 30: 0%, 31: 0%) [PENDING upload] |
| Client 3 | **43.5%** (12: 84%, 13: 1%, 14: 8%, 15: 81%) | **40.5%** (32: 93%, 33: 69%, 34: 0%, 35: 0%) |
| Client 4 | **25.8%** (16: 4%, 17: 0%, 18: 0%, 19: 99%) | **28.7%** (36: 0%, 37: 95%, 38: 18%, 39: 2%) |
