# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.83% on 6 observed labels.
Validation accuracy: 0.31333333333333335; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.418.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **80.5%** (0: 77%, 1: 84%) |
| Client 1 | **54.0%** (1: 84%, 2: 24%) |
| Client 2 | **12.0%** (2: 24%, 3: 0%) [PENDING upload] |
| Client 3 | **12.0%** (3: 0%, 4: 24%) [PENDING upload] |
| Client 4 | **12.0%** (4: 24%, 5: 0%) |
