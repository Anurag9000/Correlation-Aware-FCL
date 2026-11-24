# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.83% on 6 observed labels.
Validation accuracy: 0.22666666666666666; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.4325.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **39.0%** (0: 11%, 1: 67%) [PENDING upload] |
| Client 1 | **74.5%** (1: 67%, 2: 82%) |
| Client 2 | **41.0%** (2: 82%, 3: 0%) [PENDING upload] |
| Client 3 | **0.5%** (3: 0%, 4: 1%) [PENDING upload] |
| Client 4 | **0.5%** (4: 1%, 5: 0%) |
