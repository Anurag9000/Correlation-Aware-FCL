# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.67% on 6 observed labels.
Validation accuracy: 0.21666666666666667; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.3975.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **31.0%** (0: 3%, 1: 59%) [PENDING upload] |
| Client 1 | **34.0%** (1: 59%, 2: 9%) |
| Client 2 | **4.5%** (2: 9%, 3: 0%) [PENDING upload] |
| Client 3 | **44.5%** (3: 0%, 4: 89%) [PENDING upload] |
| Client 4 | **44.5%** (4: 89%, 5: 0%) |
