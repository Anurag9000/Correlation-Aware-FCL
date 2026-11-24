# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.83% on 6 observed labels.
Validation accuracy: 0.2633333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.346.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **82.5%** (0: 75%, 1: 90%) |
| Client 1 | **46.0%** (1: 90%, 2: 2%) |
| Client 2 | **3.5%** (2: 2%, 3: 5%) |
| Client 3 | **3.0%** (3: 5%, 4: 1%) |
| Client 4 | **0.5%** (4: 1%, 5: 0%) [PENDING upload] |
