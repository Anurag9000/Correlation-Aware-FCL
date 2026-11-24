# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.17% on 6 observed labels.
Validation accuracy: 0.4266666666666667; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.436.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **59.0%** (0: 18%, 1: 100%) |
| Client 1 | **50.0%** (1: 100%, 2: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) |
| Client 3 | **23.5%** (3: 0%, 4: 47%) |
| Client 4 | **73.5%** (4: 47%, 5: 100%) [PENDING upload] |
