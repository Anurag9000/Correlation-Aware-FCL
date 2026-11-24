# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.90% on 10 observed labels.
Validation accuracy: 0.618; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.619.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **74.0%** (0: 97%, 1: 51%) | **74.0%** (0: 97%, 1: 51%) |
| Client 1 | **70.0%** (2: 40%, 3: 100%) | **70.0%** (2: 40%, 3: 100%) |
| Client 2 | **56.0%** (4: 100%, 5: 12%) | **56.0%** (4: 100%, 5: 12%) |
| Client 3 | **57.5%** (6: 18%, 7: 97%) | **57.5%** (6: 18%, 7: 97%) |
| Client 4 | **52.0%** (8: 99%, 9: 5%) | **52.0%** (8: 99%, 9: 5%) |
