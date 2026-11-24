# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.94% on 16 observed labels.
Validation accuracy: 0.5425; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.519375.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.0%** (0: 2%, 1: 100%) | **50.5%** (8: 100%, 9: 1%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **49.0%** (10: 98%, 11: 0%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) | **50.0%** (12: 94%, 13: 6%) |
| Client 3 | **51.0%** (6: 2%, 7: 100%) | **64.0%** (14: 76%, 15: 52%) |
| Client 4 | **50.5%** (8: 100%, 9: 1%) | **50.0%** (4: 100%, 5: 0%) |
