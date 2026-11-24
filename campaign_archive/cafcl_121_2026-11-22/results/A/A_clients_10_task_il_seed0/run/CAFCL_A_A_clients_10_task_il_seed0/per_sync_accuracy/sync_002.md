# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.05% on 40 observed labels.
Validation accuracy: 0.546; seen labels never predicted: 11.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.5405.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **50.0%** (20: 0%, 21: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **74.0%** (22: 75%, 23: 73%) |
| Client 2 | **53.5%** (4: 100%, 5: 7%) | **63.0%** (24: 28%, 25: 98%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **50.0%** (26: 0%, 27: 100%) |
| Client 4 | **50.5%** (8: 100%, 9: 1%) | **50.0%** (28: 100%, 29: 0%) |
| Client 5 | **50.0%** (10: 100%, 11: 0%) | **75.5%** (30: 54%, 31: 97%) |
| Client 6 | **51.0%** (12: 98%, 13: 4%) | **50.0%** (32: 0%, 33: 100%) |
| Client 7 | **61.5%** (14: 25%, 15: 98%) | **50.0%** (34: 100%, 35: 0%) |
| Client 8 | **51.0%** (16: 2%, 17: 100%) | **50.0%** (36: 0%, 37: 100%) |
| Client 9 | **51.0%** (18: 98%, 19: 4%) | **50.0%** (38: 100%, 39: 0%) |
