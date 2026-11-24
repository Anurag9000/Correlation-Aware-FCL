# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.58% on 50 observed labels.
Validation accuracy: 0.1048; seen labels never predicted: 40.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.1058.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **10.1%** (0: 0%, 1: 0%, 2: 0%, 3: 0%, 4: 100%, 5: 0%, 6: 0%, 7: 0%, 8: 0%, 9: 1%) |
| Client 1 | **10.0%** (10: 0%, 11: 0%, 12: 0%, 13: 0%, 14: 0%, 15: 0%, 16: 0%, 17: 0%, 18: 0%, 19: 100%) |
| Client 2 | **12.8%** (20: 0%, 21: 0%, 22: 0%, 23: 28%, 24: 0%, 25: 0%, 26: 0%, 27: 100%, 28: 0%, 29: 0%) |
| Client 3 | **10.0%** (30: 0%, 31: 0%, 32: 0%, 33: 0%, 34: 0%, 35: 0%, 36: 0%, 37: 0%, 38: 100%, 39: 0%) |
| Client 4 | **10.0%** (40: 0%, 41: 0%, 42: 0%, 43: 0%, 44: 100%, 45: 0%, 46: 0%, 47: 0%, 48: 0%, 49: 0%) |
