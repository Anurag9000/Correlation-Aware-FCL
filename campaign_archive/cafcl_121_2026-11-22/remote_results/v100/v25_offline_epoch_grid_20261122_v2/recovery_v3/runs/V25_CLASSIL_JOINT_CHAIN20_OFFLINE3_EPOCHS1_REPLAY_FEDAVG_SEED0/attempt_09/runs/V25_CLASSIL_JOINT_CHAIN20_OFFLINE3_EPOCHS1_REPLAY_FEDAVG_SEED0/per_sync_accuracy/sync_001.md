# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.33% on 6 observed labels.
Validation accuracy: 0.24666666666666667; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.3975.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **29.0%** (0: 6%, 1: 52%) [PENDING upload] |
| Client 1 | **37.5%** (1: 52%, 2: 23%) |
| Client 2 | **11.5%** (2: 23%, 3: 0%) [PENDING upload] |
| Client 3 | **41.5%** (3: 0%, 4: 83%) [PENDING upload] |
| Client 4 | **41.5%** (4: 83%, 5: 0%) |
