# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.50% on 6 observed labels.
Validation accuracy: 0.27; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.43.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **25.5%** (0: 3%, 1: 48%) [PENDING upload] |
| Client 1 | **39.0%** (1: 48%, 2: 30%) |
| Client 2 | **15.0%** (2: 30%, 3: 0%) [PENDING upload] |
| Client 3 | **45.0%** (3: 0%, 4: 90%) [PENDING upload] |
| Client 4 | **45.0%** (4: 90%, 5: 0%) |
