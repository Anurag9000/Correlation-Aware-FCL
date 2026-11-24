# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.67% on 6 observed labels.
Validation accuracy: 0.2633333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.368.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **81.5%** (0: 82%, 1: 81%) |
| Client 1 | **42.5%** (1: 81%, 2: 4%) |
| Client 2 | **9.5%** (2: 4%, 3: 15%) |
| Client 3 | **8.5%** (3: 15%, 4: 2%) |
| Client 4 | **1.0%** (4: 2%, 5: 0%) [PENDING upload] |
