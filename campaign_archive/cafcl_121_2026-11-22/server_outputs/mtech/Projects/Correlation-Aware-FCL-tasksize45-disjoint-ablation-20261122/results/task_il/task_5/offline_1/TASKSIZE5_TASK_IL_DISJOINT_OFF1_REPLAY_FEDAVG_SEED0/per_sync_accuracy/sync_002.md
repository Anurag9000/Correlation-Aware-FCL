# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.72% on 50 observed labels.
Validation accuracy: 0.2864; seen labels never predicted: 14.
Server-accepted classes: 45; pending private classes: 5.
Accepted-only TEST accuracy: 0.27844444444444444.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **20.6%** (0: 0%, 1: 0%, 2: 1%, 3: 3%, 4: 99%) | **22.4%** (25: 7%, 26: 0%, 27: 100%, 28: 3%, 29: 2%) |
| Client 1 | **36.0%** (5: 53%, 6: 5%, 7: 16%, 8: 96%, 9: 10%) | **28.8%** (30: 13%, 31: 33%, 32: 0%, 33: 1%, 34: 97%) |
| Client 2 | **39.0%** (10: 46%, 11: 15%, 12: 60%, 13: 69%, 14: 5%) | **26.6%** (35: 0%, 36: 90%, 37: 38%, 38: 0%, 39: 5%) [PENDING upload] |
| Client 3 | **29.2%** (15: 1%, 16: 0%, 17: 12%, 18: 97%, 19: 36%) | **26.8%** (40: 0%, 41: 0%, 42: 59%, 43: 0%, 44: 75%) |
| Client 4 | **27.0%** (20: 2%, 21: 6%, 22: 0%, 23: 58%, 24: 69%) | **20.8%** (45: 4%, 46: 100%, 47: 0%, 48: 0%, 49: 0%) |
