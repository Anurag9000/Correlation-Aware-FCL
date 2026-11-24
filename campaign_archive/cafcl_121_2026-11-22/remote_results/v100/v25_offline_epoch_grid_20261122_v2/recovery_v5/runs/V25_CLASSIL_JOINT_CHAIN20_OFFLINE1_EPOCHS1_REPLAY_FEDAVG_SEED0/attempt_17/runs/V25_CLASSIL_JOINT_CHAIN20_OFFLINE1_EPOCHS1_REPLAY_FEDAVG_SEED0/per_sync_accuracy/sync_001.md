# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.17% on 6 observed labels.
Validation accuracy: 0.25; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.362.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **83.0%** (0: 78%, 1: 88%) |
| Client 1 | **46.5%** (1: 88%, 2: 5%) |
| Client 2 | **7.0%** (2: 5%, 3: 9%) |
| Client 3 | **5.0%** (3: 9%, 4: 1%) |
| Client 4 | **0.5%** (4: 1%, 5: 0%) [PENDING upload] |
