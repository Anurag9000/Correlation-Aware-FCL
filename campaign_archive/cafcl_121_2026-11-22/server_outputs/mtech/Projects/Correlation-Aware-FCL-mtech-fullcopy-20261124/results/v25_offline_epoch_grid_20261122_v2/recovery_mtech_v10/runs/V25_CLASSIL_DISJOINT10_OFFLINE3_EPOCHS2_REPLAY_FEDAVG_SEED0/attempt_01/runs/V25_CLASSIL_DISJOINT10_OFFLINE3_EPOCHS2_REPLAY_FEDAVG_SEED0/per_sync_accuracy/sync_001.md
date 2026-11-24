# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.80% on 10 observed labels.
Validation accuracy: 0.136; seen labels never predicted: 4.
Server-accepted classes: 4; pending private classes: 6.
Accepted-only TEST accuracy: 0.385.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **1.5%** (0: 1%, 1: 2%) [PENDING upload] |
| Client 1 | **54.5%** (2: 91%, 3: 18%) |
| Client 2 | **5.5%** (4: 11%, 5: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) [PENDING upload] |
| Client 4 | **2.5%** (8: 5%, 9: 0%) |
