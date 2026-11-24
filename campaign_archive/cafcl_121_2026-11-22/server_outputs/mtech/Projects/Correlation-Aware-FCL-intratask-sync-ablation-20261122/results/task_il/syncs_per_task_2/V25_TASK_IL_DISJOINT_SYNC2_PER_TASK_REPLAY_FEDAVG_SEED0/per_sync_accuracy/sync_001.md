# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.80% on 10 observed labels.
Validation accuracy: 0.542; seen labels never predicted: 3.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.548.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **73.5%** (4: 85%, 5: 62%) |
| Client 3 | **50.0%** (6: 100%, 7: 0%) |
| Client 4 | **50.5%** (8: 100%, 9: 1%) |
