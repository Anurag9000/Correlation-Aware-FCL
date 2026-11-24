# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.88% on 50 observed labels.
Validation accuracy: 0.2508; seen labels never predicted: 26.
Server-accepted classes: 25; pending private classes: 25.
Accepted-only TEST accuracy: 0.2688.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **18.4%** (0: 0%, 1: 34%, 2: 0%, 3: 0%, 4: 58%) [PENDING upload] | **20.0%** (25: 0%, 26: 100%, 27: 0%, 28: 0%, 29: 0%) [PENDING upload] |
| Client 1 | **30.4%** (5: 62%, 6: 0%, 7: 0%, 8: 90%, 9: 0%) | **25.8%** (30: 0%, 31: 58%, 32: 71%, 33: 0%, 34: 0%) [PENDING upload] |
| Client 2 | **27.6%** (10: 42%, 11: 2%, 12: 83%, 13: 11%, 14: 0%) | **21.0%** (35: 10%, 36: 2%, 37: 69%, 38: 0%, 39: 24%) |
| Client 3 | **20.6%** (15: 0%, 16: 0%, 17: 99%, 18: 0%, 19: 4%) [PENDING upload] | **29.6%** (40: 0%, 41: 49%, 42: 99%, 43: 0%, 44: 0%) [PENDING upload] |
| Client 4 | **21.4%** (20: 0%, 21: 0%, 22: 7%, 23: 100%, 24: 0%) | **34.0%** (45: 94%, 46: 0%, 47: 0%, 48: 0%, 49: 76%) |
