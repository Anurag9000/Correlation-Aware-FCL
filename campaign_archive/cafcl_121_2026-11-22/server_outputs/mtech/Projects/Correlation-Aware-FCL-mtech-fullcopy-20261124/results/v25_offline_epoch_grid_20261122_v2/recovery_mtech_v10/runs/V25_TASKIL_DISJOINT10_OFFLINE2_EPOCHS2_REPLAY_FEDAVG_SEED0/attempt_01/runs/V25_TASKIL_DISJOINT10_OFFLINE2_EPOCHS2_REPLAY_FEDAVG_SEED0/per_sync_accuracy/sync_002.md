# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.25% on 20 observed labels.
Validation accuracy: 0.569; seen labels never predicted: 2.
Server-accepted classes: 14; pending private classes: 6.
Accepted-only TEST accuracy: 0.6192857142857143.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **73.5%** (0: 70%, 1: 77%) | **54.5%** (10: 91%, 11: 18%) [PENDING upload] |
| Client 1 | **72.0%** (2: 45%, 3: 99%) | **50.0%** (12: 100%, 13: 0%) |
| Client 2 | **44.5%** (4: 6%, 5: 83%) [PENDING upload] | **50.0%** (14: 100%, 15: 0%) [PENDING upload] |
| Client 3 | **54.5%** (6: 14%, 7: 95%) | **69.0%** (16: 67%, 17: 71%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **63.0%** (18: 81%, 19: 45%) |
