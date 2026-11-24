# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 39.83% on 6 observed labels.
Validation accuracy: 0.3333333333333333; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.478.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **75.5%** (0: 76%, 1: 75%) |
| Client 1 | **57.5%** (1: 75%, 2: 40%) |
| Client 2 | **20.0%** (2: 40%, 3: 0%) [PENDING upload] |
| Client 3 | **24.0%** (3: 0%, 4: 48%) [PENDING upload] |
| Client 4 | **24.0%** (4: 48%, 5: 0%) |
