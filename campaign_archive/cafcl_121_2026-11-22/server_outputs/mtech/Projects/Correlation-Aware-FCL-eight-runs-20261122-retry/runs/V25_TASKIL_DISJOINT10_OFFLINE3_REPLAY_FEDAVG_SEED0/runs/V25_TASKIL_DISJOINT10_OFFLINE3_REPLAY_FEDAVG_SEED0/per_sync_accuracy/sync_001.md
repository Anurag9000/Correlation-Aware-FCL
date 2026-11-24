# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.30% on 10 observed labels.
Validation accuracy: 0.518; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 6.
Accepted-only TEST accuracy: 0.505.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **55.5%** (0: 11%, 1: 100%) [PENDING upload] |
| Client 1 | **50.5%** (2: 1%, 3: 100%) |
| Client 2 | **50.0%** (4: 0%, 5: 100%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] |
| Client 4 | **50.5%** (8: 100%, 9: 1%) |
