# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.17% on 6 observed labels.
Validation accuracy: 0.2633333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.35.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **79.0%** (0: 64%, 1: 94%) |
| Client 1 | **51.0%** (1: 94%, 2: 8%) |
| Client 2 | **7.5%** (2: 8%, 3: 7%) |
| Client 3 | **4.5%** (3: 7%, 4: 2%) |
| Client 4 | **1.0%** (4: 2%, 5: 0%) [PENDING upload] |
