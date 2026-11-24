# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.50% on 6 observed labels.
Validation accuracy: 0.2833333333333333; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.338.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **64.5%** (0: 81%, 1: 48%) |
| Client 1 | **27.5%** (1: 48%, 2: 7%) |
| Client 2 | **7.5%** (2: 7%, 3: 8%) [PENDING upload] |
| Client 3 | **14.5%** (3: 8%, 4: 21%) [PENDING upload] |
| Client 4 | **10.5%** (4: 21%, 5: 0%) |
