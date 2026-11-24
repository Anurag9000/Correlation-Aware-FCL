# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 31.00% on 6 observed labels.
Validation accuracy: 0.2733333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.372.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **82.0%** (0: 82%, 1: 82%) |
| Client 1 | **43.0%** (1: 82%, 2: 4%) |
| Client 2 | **10.0%** (2: 4%, 3: 16%) |
| Client 3 | **9.0%** (3: 16%, 4: 2%) |
| Client 4 | **1.0%** (4: 2%, 5: 0%) [PENDING upload] |
