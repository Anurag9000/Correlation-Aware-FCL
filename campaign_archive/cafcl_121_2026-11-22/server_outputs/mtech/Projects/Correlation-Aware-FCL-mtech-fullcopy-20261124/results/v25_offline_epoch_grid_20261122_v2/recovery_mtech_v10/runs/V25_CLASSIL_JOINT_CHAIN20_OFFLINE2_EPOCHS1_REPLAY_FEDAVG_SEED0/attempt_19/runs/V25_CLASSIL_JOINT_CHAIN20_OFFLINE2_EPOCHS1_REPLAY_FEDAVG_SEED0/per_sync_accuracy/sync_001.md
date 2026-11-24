# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.00% on 6 observed labels.
Validation accuracy: 0.33666666666666667; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.42.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **77.0%** (0: 74%, 1: 80%) |
| Client 1 | **55.5%** (1: 80%, 2: 31%) |
| Client 2 | **15.5%** (2: 31%, 3: 0%) [PENDING upload] |
| Client 3 | **12.5%** (3: 0%, 4: 25%) [PENDING upload] |
| Client 4 | **12.5%** (4: 25%, 5: 0%) |
