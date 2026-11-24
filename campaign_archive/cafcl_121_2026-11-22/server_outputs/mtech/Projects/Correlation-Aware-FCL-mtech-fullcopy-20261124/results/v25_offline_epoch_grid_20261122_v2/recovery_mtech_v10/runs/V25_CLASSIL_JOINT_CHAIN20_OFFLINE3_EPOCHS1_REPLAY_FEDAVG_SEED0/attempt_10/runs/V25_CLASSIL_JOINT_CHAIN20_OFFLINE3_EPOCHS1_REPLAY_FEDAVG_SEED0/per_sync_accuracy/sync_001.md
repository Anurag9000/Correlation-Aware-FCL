# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.17% on 6 observed labels.
Validation accuracy: 0.23666666666666666; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.3975.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **26.5%** (0: 10%, 1: 43%) [PENDING upload] |
| Client 1 | **37.5%** (1: 43%, 2: 32%) |
| Client 2 | **16.0%** (2: 32%, 3: 0%) [PENDING upload] |
| Client 3 | **42.0%** (3: 0%, 4: 84%) [PENDING upload] |
| Client 4 | **42.0%** (4: 84%, 5: 0%) |
