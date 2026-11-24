# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.60% on 10 observed labels.
Validation accuracy: 0.584; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.566.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **65.5%** (0: 69%, 1: 62%) | **65.5%** (0: 69%, 1: 62%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **66.5%** (4: 93%, 5: 40%) | **66.5%** (4: 93%, 5: 40%) |
| Client 3 | **51.0%** (6: 2%, 7: 100%) | **51.0%** (6: 2%, 7: 100%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) | **50.0%** (8: 100%, 9: 0%) |
