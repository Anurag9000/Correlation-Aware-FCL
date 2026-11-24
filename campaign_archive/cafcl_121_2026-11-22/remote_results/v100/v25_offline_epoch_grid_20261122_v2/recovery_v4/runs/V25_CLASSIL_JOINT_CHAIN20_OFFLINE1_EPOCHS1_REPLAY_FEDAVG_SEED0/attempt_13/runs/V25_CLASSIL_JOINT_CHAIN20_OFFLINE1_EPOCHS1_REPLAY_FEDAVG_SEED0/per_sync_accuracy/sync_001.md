# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.67% on 6 observed labels.
Validation accuracy: 0.28; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.356.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **83.0%** (0: 87%, 1: 79%) |
| Client 1 | **40.0%** (1: 79%, 2: 1%) |
| Client 2 | **6.0%** (2: 1%, 3: 11%) |
| Client 3 | **5.5%** (3: 11%, 4: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
