# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 36.67% on 6 observed labels.
Validation accuracy: 0.31666666666666665; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.44.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **76.0%** (0: 79%, 1: 73%) |
| Client 1 | **50.0%** (1: 73%, 2: 27%) |
| Client 2 | **13.5%** (2: 27%, 3: 0%) [PENDING upload] |
| Client 3 | **20.5%** (3: 0%, 4: 41%) [PENDING upload] |
| Client 4 | **20.5%** (4: 41%, 5: 0%) |
