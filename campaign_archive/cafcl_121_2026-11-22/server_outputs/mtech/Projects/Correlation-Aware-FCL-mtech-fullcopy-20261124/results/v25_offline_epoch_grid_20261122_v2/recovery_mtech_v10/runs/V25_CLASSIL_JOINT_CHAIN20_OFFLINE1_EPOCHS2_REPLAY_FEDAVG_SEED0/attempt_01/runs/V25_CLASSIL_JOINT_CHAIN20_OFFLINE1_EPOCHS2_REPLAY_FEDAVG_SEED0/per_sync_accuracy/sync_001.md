# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 33.50% on 6 observed labels.
Validation accuracy: 0.3333333333333333; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.402.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **62.5%** (0: 64%, 1: 61%) |
| Client 1 | **39.5%** (1: 61%, 2: 18%) |
| Client 2 | **9.0%** (2: 18%, 3: 0%) |
| Client 3 | **29.0%** (3: 0%, 4: 58%) |
| Client 4 | **29.0%** (4: 58%, 5: 0%) [PENDING upload] |
