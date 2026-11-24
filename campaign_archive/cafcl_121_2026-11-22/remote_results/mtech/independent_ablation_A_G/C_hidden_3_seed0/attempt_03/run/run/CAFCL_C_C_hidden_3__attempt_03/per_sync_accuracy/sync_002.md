# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.90% on 20 observed labels.
Validation accuracy: 0.592; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.628125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.0%** (8: 99%, 9: 3%) | **58.5%** (14: 60%, 15: 57%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **66.5%** (18: 38%, 19: 95%) [PENDING upload] |
| Client 2 | **70.5%** (4: 87%, 5: 54%) | **71.0%** (16: 43%, 17: 99%) |
| Client 3 | **79.0%** (0: 63%, 1: 95%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **72.5%** (2: 45%, 3: 100%) | **50.0%** (12: 100%, 13: 0%) |
