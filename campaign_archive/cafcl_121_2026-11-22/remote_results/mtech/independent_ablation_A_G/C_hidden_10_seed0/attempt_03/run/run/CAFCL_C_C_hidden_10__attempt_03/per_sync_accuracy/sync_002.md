# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.90% on 20 observed labels.
Validation accuracy: 0.596; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.62.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.5%** (8: 99%, 9: 4%) | **64.5%** (14: 77%, 15: 52%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **53.0%** (18: 25%, 19: 81%) [PENDING upload] |
| Client 2 | **74.0%** (4: 89%, 5: 59%) | **60.0%** (16: 20%, 17: 100%) |
| Client 3 | **71.0%** (0: 46%, 1: 96%) | **51.0%** (10: 95%, 11: 7%) |
| Client 4 | **72.5%** (2: 45%, 3: 100%) | **51.5%** (12: 99%, 13: 4%) |
