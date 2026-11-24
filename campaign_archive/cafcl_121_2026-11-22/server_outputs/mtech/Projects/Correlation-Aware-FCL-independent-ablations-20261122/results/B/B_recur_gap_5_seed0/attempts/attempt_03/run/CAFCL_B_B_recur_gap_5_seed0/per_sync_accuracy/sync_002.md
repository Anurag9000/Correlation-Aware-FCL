# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.20% on 20 observed labels.
Validation accuracy: 0.592; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.592.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **65.5%** (0: 31%, 1: 100%) | **54.0%** (10: 94%, 11: 14%) |
| Client 1 | **51.0%** (2: 2%, 3: 100%) | **50.0%** (12: 0%, 13: 100%) |
| Client 2 | **73.0%** (4: 89%, 5: 57%) | **65.0%** (14: 72%, 15: 58%) |
| Client 3 | **50.5%** (6: 2%, 7: 99%) | **81.0%** (16: 81%, 17: 81%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **50.5%** (18: 18%, 19: 83%) |
