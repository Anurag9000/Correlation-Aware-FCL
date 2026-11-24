# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 47.23% on 30 observed labels.
Validation accuracy: 0.4786666666666667; seen labels never predicted: 6.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.4723333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **47.7%** (0: 1%, 1: 44%, 2: 98%) | **34.3%** (15: 100%, 16: 0%, 17: 3%) |
| Client 1 | **55.7%** (3: 4%, 4: 91%, 5: 72%) | **46.0%** (18: 85%, 19: 53%, 20: 0%) |
| Client 2 | **54.0%** (6: 0%, 7: 85%, 8: 77%) | **67.3%** (21: 72%, 22: 35%, 23: 95%) |
| Client 3 | **37.0%** (9: 10%, 10: 99%, 11: 2%) | **43.0%** (24: 0%, 25: 60%, 26: 69%) |
| Client 4 | **54.0%** (12: 86%, 13: 7%, 14: 69%) | **33.3%** (27: 100%, 28: 0%, 29: 0%) |
