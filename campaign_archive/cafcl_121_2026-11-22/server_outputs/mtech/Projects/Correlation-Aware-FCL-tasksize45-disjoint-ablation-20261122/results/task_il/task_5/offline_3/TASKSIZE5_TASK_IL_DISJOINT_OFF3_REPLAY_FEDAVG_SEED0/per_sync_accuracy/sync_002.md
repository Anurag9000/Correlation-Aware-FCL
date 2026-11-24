# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.90% on 50 observed labels.
Validation accuracy: 0.2452; seen labels never predicted: 23.
Server-accepted classes: 25; pending private classes: 25.
Accepted-only TEST accuracy: 0.2724.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **15.4%** (0: 53%, 1: 24%, 2: 0%, 3: 0%, 4: 0%) [PENDING upload] | **20.8%** (25: 7%, 26: 89%, 27: 0%, 28: 0%, 29: 8%) [PENDING upload] |
| Client 1 | **23.6%** (5: 0%, 6: 18%, 7: 0%, 8: 100%, 9: 0%) | **20.0%** (30: 0%, 31: 0%, 32: 100%, 33: 0%, 34: 0%) [PENDING upload] |
| Client 2 | **26.0%** (10: 95%, 11: 3%, 12: 30%, 13: 2%, 14: 0%) | **23.0%** (35: 0%, 36: 98%, 37: 17%, 38: 0%, 39: 0%) |
| Client 3 | **26.8%** (15: 0%, 16: 0%, 17: 70%, 18: 0%, 19: 64%) [PENDING upload] | **19.8%** (40: 65%, 41: 34%, 42: 0%, 43: 0%, 44: 0%) [PENDING upload] |
| Client 4 | **38.6%** (20: 0%, 21: 96%, 22: 66%, 23: 31%, 24: 0%) | **25.0%** (45: 100%, 46: 0%, 47: 0%, 48: 0%, 49: 25%) |
