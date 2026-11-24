# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.83% on 6 observed labels.
Validation accuracy: 0.4033333333333333; seen labels never predicted: 0.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.426.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **55.0%** (0: 100%, 1: 10%) |
| Client 1 | **6.5%** (1: 10%, 2: 3%) |
| Client 2 | **1.5%** (2: 3%, 3: 0%) |
| Client 3 | **28.0%** (3: 0%, 4: 56%) |
| Client 4 | **78.0%** (4: 56%, 5: 100%) [PENDING upload] |
