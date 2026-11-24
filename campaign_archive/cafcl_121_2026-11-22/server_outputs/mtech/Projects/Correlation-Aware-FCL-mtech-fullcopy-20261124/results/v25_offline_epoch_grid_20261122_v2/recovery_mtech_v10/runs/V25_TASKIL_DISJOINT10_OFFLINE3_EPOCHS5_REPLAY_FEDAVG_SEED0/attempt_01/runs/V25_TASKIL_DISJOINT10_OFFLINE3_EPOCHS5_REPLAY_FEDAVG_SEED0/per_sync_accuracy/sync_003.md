# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.90% on 30 observed labels.
Validation accuracy: 0.5293333333333333; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.517.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **63.5%** (0: 32%, 1: 95%) [PENDING upload] | **58.5%** (10: 76%, 11: 41%) [PENDING upload] | **54.0%** (20: 8%, 21: 100%) [PENDING upload] |
| Client 1 | **63.0%** (2: 26%, 3: 100%) | **50.0%** (12: 100%, 13: 0%) | **49.5%** (22: 98%, 23: 1%) |
| Client 2 | **50.5%** (4: 100%, 5: 1%) | **50.0%** (14: 0%, 15: 100%) | **50.0%** (24: 0%, 25: 100%) [PENDING upload] |
| Client 3 | **52.5%** (6: 6%, 7: 99%) | **50.0%** (16: 100%, 17: 0%) | **50.0%** (26: 100%, 27: 0%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **50.0%** (18: 1%, 19: 99%) | **50.5%** (28: 11%, 29: 90%) [PENDING upload] |
