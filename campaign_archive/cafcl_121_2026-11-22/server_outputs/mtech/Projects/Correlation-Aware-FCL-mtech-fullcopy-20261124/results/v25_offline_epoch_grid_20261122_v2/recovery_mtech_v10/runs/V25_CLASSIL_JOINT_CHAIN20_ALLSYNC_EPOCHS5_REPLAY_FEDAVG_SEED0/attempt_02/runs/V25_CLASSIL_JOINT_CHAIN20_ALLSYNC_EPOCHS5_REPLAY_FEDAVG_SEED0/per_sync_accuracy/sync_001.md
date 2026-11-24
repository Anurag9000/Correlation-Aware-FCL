# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.00% on 6 observed labels.
Validation accuracy: 0.25666666666666665; seen labels never predicted: 4.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.25.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **30.0%** (0: 60%, 1: 0%) |
| Client 1 | **0.0%** (1: 0%, 2: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) |
| Client 3 | **45.0%** (3: 0%, 4: 90%) |
| Client 4 | **45.0%** (4: 90%, 5: 0%) |
