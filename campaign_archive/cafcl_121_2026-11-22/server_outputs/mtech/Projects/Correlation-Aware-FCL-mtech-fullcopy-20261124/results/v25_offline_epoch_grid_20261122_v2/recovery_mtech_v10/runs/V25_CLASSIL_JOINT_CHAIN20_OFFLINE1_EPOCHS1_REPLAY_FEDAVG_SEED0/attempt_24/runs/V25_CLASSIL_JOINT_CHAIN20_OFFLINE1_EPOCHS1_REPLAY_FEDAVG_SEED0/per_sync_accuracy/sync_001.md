# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.67% on 6 observed labels.
Validation accuracy: 0.25333333333333335; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.344.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **81.5%** (0: 77%, 1: 86%) |
| Client 1 | **44.0%** (1: 86%, 2: 2%) |
| Client 2 | **3.5%** (2: 2%, 3: 5%) |
| Client 3 | **3.5%** (3: 5%, 4: 2%) |
| Client 4 | **1.0%** (4: 2%, 5: 0%) [PENDING upload] |
