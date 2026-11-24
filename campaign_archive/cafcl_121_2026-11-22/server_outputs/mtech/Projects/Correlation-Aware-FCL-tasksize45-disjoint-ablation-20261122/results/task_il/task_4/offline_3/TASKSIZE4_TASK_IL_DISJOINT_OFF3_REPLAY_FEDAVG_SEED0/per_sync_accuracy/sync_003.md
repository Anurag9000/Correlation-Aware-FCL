# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.98% on 60 observed labels.
Validation accuracy: 0.2663333333333333; seen labels never predicted: 37.
Server-accepted classes: 40; pending private classes: 20.
Accepted-only TEST accuracy: 0.27975.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **25.0%** (0: 100%, 1: 0%, 2: 0%, 3: 0%) [PENDING upload] | **25.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] | **25.0%** (40: 0%, 41: 100%, 42: 0%, 43: 0%) [PENDING upload] |
| Client 1 | **25.0%** (4: 100%, 5: 0%, 6: 0%, 7: 0%) | **26.8%** (24: 0%, 25: 7%, 26: 0%, 27: 100%) | **25.0%** (44: 100%, 45: 0%, 46: 0%, 47: 0%) |
| Client 2 | **29.0%** (8: 94%, 9: 0%, 10: 0%, 11: 22%) | **23.8%** (28: 0%, 29: 11%, 30: 0%, 31: 84%) | **25.0%** (48: 0%, 49: 0%, 50: 0%, 51: 100%) [PENDING upload] |
| Client 3 | **29.8%** (12: 0%, 13: 0%, 14: 32%, 15: 87%) | **26.5%** (32: 0%, 33: 0%, 34: 93%, 35: 13%) | **25.0%** (52: 0%, 53: 0%, 54: 0%, 55: 100%) |
| Client 4 | **31.2%** (16: 0%, 17: 0%, 18: 25%, 19: 100%) | **37.8%** (36: 0%, 37: 0%, 38: 89%, 39: 62%) | **25.0%** (56: 0%, 57: 0%, 58: 0%, 59: 100%) [PENDING upload] |
