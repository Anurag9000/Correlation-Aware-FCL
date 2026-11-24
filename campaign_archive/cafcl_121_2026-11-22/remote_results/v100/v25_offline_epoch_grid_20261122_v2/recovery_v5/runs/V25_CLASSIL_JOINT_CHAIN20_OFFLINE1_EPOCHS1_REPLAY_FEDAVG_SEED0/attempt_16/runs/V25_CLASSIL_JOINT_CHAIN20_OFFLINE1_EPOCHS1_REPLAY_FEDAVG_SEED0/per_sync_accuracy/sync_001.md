# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.33% on 6 observed labels.
Validation accuracy: 0.26666666666666666; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.352.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **83.0%** (0: 76%, 1: 90%) |
| Client 1 | **49.0%** (1: 90%, 2: 8%) |
| Client 2 | **4.5%** (2: 8%, 3: 1%) |
| Client 3 | **1.0%** (3: 1%, 4: 1%) |
| Client 4 | **0.5%** (4: 1%, 5: 0%) [PENDING upload] |
