# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.17% on 6 observed labels.
Validation accuracy: 0.31333333333333335; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.41.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **77.0%** (0: 81%, 1: 73%) |
| Client 1 | **43.5%** (1: 73%, 2: 14%) |
| Client 2 | **7.0%** (2: 14%, 3: 0%) [PENDING upload] |
| Client 3 | **18.5%** (3: 0%, 4: 37%) [PENDING upload] |
| Client 4 | **18.5%** (4: 37%, 5: 0%) |
