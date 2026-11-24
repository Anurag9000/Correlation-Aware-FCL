# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 47.00% on 6 observed labels.
Validation accuracy: 0.45666666666666667; seen labels never predicted: 0.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.6875.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 3%, 1: 97%) [PENDING upload] |
| Client 1 | **68.5%** (1: 97%, 2: 40%) |
| Client 2 | **20.0%** (2: 40%, 3: 0%) [PENDING upload] |
| Client 3 | **22.0%** (3: 0%, 4: 44%) [PENDING upload] |
| Client 4 | **71.0%** (4: 44%, 5: 98%) |
