# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.50% on 6 observed labels.
Validation accuracy: 0.43; seen labels never predicted: 0.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.675.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **51.0%** (0: 14%, 1: 88%) [PENDING upload] |
| Client 1 | **58.0%** (1: 88%, 2: 28%) |
| Client 2 | **14.0%** (2: 28%, 3: 0%) [PENDING upload] |
| Client 3 | **49.0%** (3: 0%, 4: 98%) [PENDING upload] |
| Client 4 | **59.5%** (4: 98%, 5: 21%) |
