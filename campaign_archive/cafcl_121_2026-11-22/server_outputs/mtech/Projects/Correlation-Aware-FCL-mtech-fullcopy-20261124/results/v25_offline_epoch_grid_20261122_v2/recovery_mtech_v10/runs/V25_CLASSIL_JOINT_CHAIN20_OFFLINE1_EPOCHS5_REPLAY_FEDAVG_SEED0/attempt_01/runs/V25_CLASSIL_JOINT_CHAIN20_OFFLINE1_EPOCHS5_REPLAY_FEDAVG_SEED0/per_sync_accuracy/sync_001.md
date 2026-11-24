# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.83% on 6 observed labels.
Validation accuracy: 0.2633333333333333; seen labels never predicted: 4.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.298.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **29.5%** (0: 59%, 1: 0%) |
| Client 1 | **0.0%** (1: 0%, 2: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) |
| Client 3 | **45.0%** (3: 0%, 4: 90%) |
| Client 4 | **45.0%** (4: 90%, 5: 0%) [PENDING upload] |
