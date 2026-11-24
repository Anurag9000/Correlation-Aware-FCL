# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.83% on 6 observed labels.
Validation accuracy: 0.31333333333333335; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.43.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **78.0%** (0: 81%, 1: 75%) |
| Client 1 | **51.0%** (1: 75%, 2: 27%) |
| Client 2 | **13.5%** (2: 27%, 3: 0%) [PENDING upload] |
| Client 3 | **16.0%** (3: 0%, 4: 32%) [PENDING upload] |
| Client 4 | **16.0%** (4: 32%, 5: 0%) |
