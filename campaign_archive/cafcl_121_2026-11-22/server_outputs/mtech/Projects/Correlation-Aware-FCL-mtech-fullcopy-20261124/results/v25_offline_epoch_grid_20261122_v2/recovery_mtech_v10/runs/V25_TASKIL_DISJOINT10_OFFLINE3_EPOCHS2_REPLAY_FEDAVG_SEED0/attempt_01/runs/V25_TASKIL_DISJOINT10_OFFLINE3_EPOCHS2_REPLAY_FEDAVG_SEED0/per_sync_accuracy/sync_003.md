# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 49.40% on 30 observed labels.
Validation accuracy: 0.49866666666666665; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.485.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **50.0%** (0: 1%, 1: 99%) [PENDING upload] | **56.0%** (10: 15%, 11: 97%) [PENDING upload] | **50.0%** (20: 100%, 21: 0%) [PENDING upload] |
| Client 1 | **55.0%** (2: 93%, 3: 17%) | **50.5%** (12: 2%, 13: 99%) | **26.0%** (22: 30%, 23: 22%) |
| Client 2 | **49.5%** (4: 4%, 5: 95%) | **50.0%** (14: 100%, 15: 0%) | **53.0%** (24: 97%, 25: 9%) [PENDING upload] |
| Client 3 | **56.5%** (6: 98%, 7: 15%) | **50.0%** (16: 100%, 17: 0%) | **50.0%** (26: 100%, 27: 0%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) | **47.5%** (18: 90%, 19: 5%) | **47.0%** (28: 92%, 29: 2%) [PENDING upload] |
