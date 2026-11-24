# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 39.83% on 6 observed labels.
Validation accuracy: 0.41333333333333333; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.506.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **69.5%** (0: 99%, 1: 40%) |
| Client 1 | **20.0%** (1: 40%, 2: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) [PENDING upload] |
| Client 3 | **50.0%** (3: 0%, 4: 100%) [PENDING upload] |
| Client 4 | **50.0%** (4: 100%, 5: 0%) |
