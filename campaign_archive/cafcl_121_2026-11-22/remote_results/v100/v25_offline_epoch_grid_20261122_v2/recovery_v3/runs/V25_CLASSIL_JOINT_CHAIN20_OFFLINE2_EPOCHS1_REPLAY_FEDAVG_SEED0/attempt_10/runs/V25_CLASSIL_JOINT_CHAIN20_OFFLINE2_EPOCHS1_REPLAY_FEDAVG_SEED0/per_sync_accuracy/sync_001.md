# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.67% on 6 observed labels.
Validation accuracy: 0.32666666666666666; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.428.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **77.0%** (0: 81%, 1: 73%) |
| Client 1 | **45.5%** (1: 73%, 2: 18%) |
| Client 2 | **9.0%** (2: 18%, 3: 0%) [PENDING upload] |
| Client 3 | **21.0%** (3: 0%, 4: 42%) [PENDING upload] |
| Client 4 | **21.0%** (4: 42%, 5: 0%) |
