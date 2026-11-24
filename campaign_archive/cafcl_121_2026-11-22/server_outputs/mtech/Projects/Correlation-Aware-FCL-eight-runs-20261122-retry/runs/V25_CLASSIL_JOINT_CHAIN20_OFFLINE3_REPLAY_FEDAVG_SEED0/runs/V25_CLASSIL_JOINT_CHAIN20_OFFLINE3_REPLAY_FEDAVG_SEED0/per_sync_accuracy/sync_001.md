# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.50% on 6 observed labels.
Validation accuracy: 0.21333333333333335; seen labels never predicted: 1.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.3525.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **40.5%** (0: 1%, 1: 80%) [PENDING upload] |
| Client 1 | **63.0%** (1: 80%, 2: 46%) |
| Client 2 | **23.0%** (2: 46%, 3: 0%) [PENDING upload] |
| Client 3 | **3.5%** (3: 0%, 4: 7%) [PENDING upload] |
| Client 4 | **7.0%** (4: 7%, 5: 7%) |
