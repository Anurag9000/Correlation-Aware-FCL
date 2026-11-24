# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.33% on 6 observed labels.
Validation accuracy: 0.45; seen labels never predicted: 0.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.44333333333333336.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **78.0%** (0: 90%, 1: 66%) |
| Client 1 | **33.0%** (1: 66%, 2: 0%) |
| Client 2 | **3.5%** (2: 0%, 3: 7%) |
| Client 3 | **53.5%** (3: 7%, 4: 100%) |
| Client 4 | **51.5%** (4: 100%, 5: 3%) |
