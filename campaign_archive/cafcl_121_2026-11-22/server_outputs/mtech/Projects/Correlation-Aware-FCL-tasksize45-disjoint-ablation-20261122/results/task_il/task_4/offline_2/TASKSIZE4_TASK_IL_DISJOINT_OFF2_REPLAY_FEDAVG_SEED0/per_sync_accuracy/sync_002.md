# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.93% on 40 observed labels.
Validation accuracy: 0.2795; seen labels never predicted: 15.
Server-accepted classes: 28; pending private classes: 12.
Accepted-only TEST accuracy: 0.3053571428571429.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **31.5%** (0: 0%, 1: 0%, 2: 26%, 3: 100%) | **25.0%** (20: 100%, 21: 0%, 22: 0%, 23: 0%) [PENDING upload] |
| Client 1 | **27.3%** (4: 100%, 5: 7%, 6: 0%, 7: 2%) | **34.5%** (24: 0%, 25: 4%, 26: 35%, 27: 99%) |
| Client 2 | **25.5%** (8: 0%, 9: 0%, 10: 55%, 11: 47%) [PENDING upload] | **25.0%** (28: 100%, 29: 0%, 30: 0%, 31: 0%) [PENDING upload] |
| Client 3 | **25.2%** (12: 0%, 13: 0%, 14: 1%, 15: 100%) | **25.2%** (32: 1%, 33: 1%, 34: 97%, 35: 2%) |
| Client 4 | **38.8%** (16: 5%, 17: 1%, 18: 62%, 19: 87%) | **31.2%** (36: 0%, 37: 1%, 38: 98%, 39: 26%) |
