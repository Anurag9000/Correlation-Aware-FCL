# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.70% on 10 observed labels.
Validation accuracy: 0.578; seen labels never predicted: 1.
Server-accepted classes: 4; pending private classes: 6.
Accepted-only TEST accuracy: 0.7325.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **51.5%** (0: 4%, 1: 99%) [PENDING upload] |
| Client 1 | **82.0%** (2: 91%, 3: 73%) |
| Client 2 | **35.5%** (4: 5%, 5: 66%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] |
| Client 4 | **64.5%** (8: 93%, 9: 36%) |
