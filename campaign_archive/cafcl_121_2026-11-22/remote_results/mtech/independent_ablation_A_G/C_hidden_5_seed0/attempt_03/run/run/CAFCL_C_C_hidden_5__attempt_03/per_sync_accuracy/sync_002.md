# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.00% on 20 observed labels.
Validation accuracy: 0.603; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.6125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.5%** (8: 100%, 9: 1%) | **68.5%** (14: 57%, 15: 80%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] |
| Client 2 | **74.0%** (4: 89%, 5: 59%) | **72.0%** (16: 45%, 17: 99%) |
| Client 3 | **50.0%** (0: 2%, 1: 98%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **59.5%** (2: 19%, 3: 100%) | **65.5%** (12: 46%, 13: 85%) |
