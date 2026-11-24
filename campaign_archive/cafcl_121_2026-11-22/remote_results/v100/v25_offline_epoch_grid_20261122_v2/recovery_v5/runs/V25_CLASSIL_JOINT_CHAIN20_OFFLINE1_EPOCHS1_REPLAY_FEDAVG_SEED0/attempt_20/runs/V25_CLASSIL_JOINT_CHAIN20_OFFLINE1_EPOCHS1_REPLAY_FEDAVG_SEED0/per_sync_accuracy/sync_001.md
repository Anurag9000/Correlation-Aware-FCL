# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 31.17% on 6 observed labels.
Validation accuracy: 0.25333333333333335; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.374.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **83.5%** (0: 76%, 1: 91%) |
| Client 1 | **47.5%** (1: 91%, 2: 4%) |
| Client 2 | **8.5%** (2: 4%, 3: 13%) |
| Client 3 | **8.0%** (3: 13%, 4: 3%) |
| Client 4 | **1.5%** (4: 3%, 5: 0%) [PENDING upload] |
