# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.83% on 6 observed labels.
Validation accuracy: 0.5866666666666667; seen labels never predicted: 0.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.5983333333333334.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **46.5%** (0: 77%, 1: 16%) |
| Client 1 | **53.5%** (1: 16%, 2: 91%) |
| Client 2 | **45.5%** (2: 91%, 3: 0%) |
| Client 3 | **37.5%** (3: 0%, 4: 75%) |
| Client 4 | **87.5%** (4: 75%, 5: 100%) |
