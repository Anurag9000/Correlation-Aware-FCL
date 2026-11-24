# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.00% on 6 observed labels.
Validation accuracy: 0.2733333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.3.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **58.0%** (0: 95%, 1: 21%) |
| Client 1 | **10.5%** (1: 21%, 2: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) [PENDING upload] |
| Client 3 | **17.0%** (3: 0%, 4: 34%) [PENDING upload] |
| Client 4 | **17.0%** (4: 34%, 5: 0%) |
