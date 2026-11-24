# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.33% on 6 observed labels.
Validation accuracy: 0.32666666666666666; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.424.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **78.5%** (0: 86%, 1: 71%) |
| Client 1 | **44.5%** (1: 71%, 2: 18%) |
| Client 2 | **9.0%** (2: 18%, 3: 0%) [PENDING upload] |
| Client 3 | **18.5%** (3: 0%, 4: 37%) [PENDING upload] |
| Client 4 | **18.5%** (4: 37%, 5: 0%) |
