# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 36.33% on 6 observed labels.
Validation accuracy: 0.31666666666666665; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.436.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **71.5%** (0: 83%, 1: 60%) |
| Client 1 | **46.0%** (1: 60%, 2: 32%) |
| Client 2 | **16.0%** (2: 32%, 3: 0%) [PENDING upload] |
| Client 3 | **21.5%** (3: 0%, 4: 43%) [PENDING upload] |
| Client 4 | **21.5%** (4: 43%, 5: 0%) |
