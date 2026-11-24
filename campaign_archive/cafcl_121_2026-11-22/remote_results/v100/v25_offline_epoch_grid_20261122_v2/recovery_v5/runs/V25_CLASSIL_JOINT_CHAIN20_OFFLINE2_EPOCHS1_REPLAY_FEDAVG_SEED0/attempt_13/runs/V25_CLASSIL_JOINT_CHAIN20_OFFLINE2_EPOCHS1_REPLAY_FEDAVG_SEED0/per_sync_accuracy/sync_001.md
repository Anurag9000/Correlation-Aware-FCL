# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 38.17% on 6 observed labels.
Validation accuracy: 0.3233333333333333; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.458.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **81.0%** (0: 83%, 1: 79%) |
| Client 1 | **51.0%** (1: 79%, 2: 23%) |
| Client 2 | **11.5%** (2: 23%, 3: 0%) [PENDING upload] |
| Client 3 | **22.0%** (3: 0%, 4: 44%) [PENDING upload] |
| Client 4 | **22.0%** (4: 44%, 5: 0%) |
