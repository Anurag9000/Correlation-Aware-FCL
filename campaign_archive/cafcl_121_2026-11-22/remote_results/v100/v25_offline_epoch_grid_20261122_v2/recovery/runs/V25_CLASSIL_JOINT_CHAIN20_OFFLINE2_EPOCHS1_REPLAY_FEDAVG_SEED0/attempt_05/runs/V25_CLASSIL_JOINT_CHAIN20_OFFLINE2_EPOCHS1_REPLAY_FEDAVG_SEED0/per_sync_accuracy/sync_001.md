# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 37.67% on 6 observed labels.
Validation accuracy: 0.3333333333333333; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.452.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **72.0%** (0: 83%, 1: 61%) |
| Client 1 | **48.0%** (1: 61%, 2: 35%) |
| Client 2 | **17.5%** (2: 35%, 3: 0%) [PENDING upload] |
| Client 3 | **23.5%** (3: 0%, 4: 47%) [PENDING upload] |
| Client 4 | **23.5%** (4: 47%, 5: 0%) |
