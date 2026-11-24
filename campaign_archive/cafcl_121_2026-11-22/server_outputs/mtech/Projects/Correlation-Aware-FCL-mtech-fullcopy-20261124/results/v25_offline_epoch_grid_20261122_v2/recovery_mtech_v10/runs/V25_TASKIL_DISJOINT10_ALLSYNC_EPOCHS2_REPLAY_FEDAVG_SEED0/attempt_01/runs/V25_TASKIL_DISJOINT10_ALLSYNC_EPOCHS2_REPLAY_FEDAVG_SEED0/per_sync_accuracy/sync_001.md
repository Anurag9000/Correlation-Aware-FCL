# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.40% on 10 observed labels.
Validation accuracy: 0.534; seen labels never predicted: 3.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.534.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **66.0%** (0: 33%, 1: 99%) |
| Client 1 | **50.0%** (2: 100%, 3: 0%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) |
| Client 3 | **51.0%** (6: 3%, 7: 99%) |
| Client 4 | **50.0%** (8: 0%, 9: 100%) |
