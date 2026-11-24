# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.18% on 50 observed labels.
Validation accuracy: 0.2772; seen labels never predicted: 18.
Server-accepted classes: 35; pending private classes: 15.
Accepted-only TEST accuracy: 0.29228571428571426.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **22.2%** (0: 0%, 1: 0%, 2: 11%, 3: 0%, 4: 100%) | **19.4%** (25: 0%, 26: 1%, 27: 0%, 28: 0%, 29: 96%) [PENDING upload] |
| Client 1 | **26.6%** (5: 33%, 6: 0%, 7: 1%, 8: 99%, 9: 0%) | **44.4%** (30: 81%, 31: 88%, 32: 0%, 33: 7%, 34: 46%) |
| Client 2 | **22.2%** (10: 0%, 11: 16%, 12: 1%, 13: 0%, 14: 94%) [PENDING upload] | **25.6%** (35: 0%, 36: 81%, 37: 43%, 38: 0%, 39: 4%) [PENDING upload] |
| Client 3 | **27.0%** (15: 1%, 16: 0%, 17: 12%, 18: 70%, 19: 52%) | **20.2%** (40: 1%, 41: 0%, 42: 100%, 43: 0%, 44: 0%) |
| Client 4 | **32.8%** (20: 0%, 21: 11%, 22: 84%, 23: 69%, 24: 0%) | **31.4%** (45: 52%, 46: 71%, 47: 14%, 48: 0%, 49: 20%) |
