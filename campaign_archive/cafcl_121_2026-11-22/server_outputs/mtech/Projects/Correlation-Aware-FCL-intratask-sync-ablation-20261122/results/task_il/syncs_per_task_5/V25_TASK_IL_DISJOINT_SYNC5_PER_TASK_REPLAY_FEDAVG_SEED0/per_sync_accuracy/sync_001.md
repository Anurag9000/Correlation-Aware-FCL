# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.60% on 10 observed labels.
Validation accuracy: 0.548; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.526.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **60.0%** (0: 29%, 1: 91%) |
| Client 1 | **50.0%** (2: 100%, 3: 0%) |
| Client 2 | **50.5%** (4: 100%, 5: 1%) |
| Client 3 | **52.5%** (6: 6%, 7: 99%) |
| Client 4 | **50.0%** (8: 0%, 9: 100%) |
