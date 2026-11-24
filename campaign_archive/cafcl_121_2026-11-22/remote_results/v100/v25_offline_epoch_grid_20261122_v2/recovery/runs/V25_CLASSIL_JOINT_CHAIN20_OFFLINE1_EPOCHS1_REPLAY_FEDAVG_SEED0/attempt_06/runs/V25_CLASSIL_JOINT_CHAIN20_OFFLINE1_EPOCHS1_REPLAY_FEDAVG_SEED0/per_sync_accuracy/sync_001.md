# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.00% on 6 observed labels.
Validation accuracy: 0.2733333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.36.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **85.0%** (0: 80%, 1: 90%) |
| Client 1 | **45.0%** (1: 90%, 2: 0%) |
| Client 2 | **4.5%** (2: 0%, 3: 9%) |
| Client 3 | **5.0%** (3: 9%, 4: 1%) |
| Client 4 | **0.5%** (4: 1%, 5: 0%) [PENDING upload] |
