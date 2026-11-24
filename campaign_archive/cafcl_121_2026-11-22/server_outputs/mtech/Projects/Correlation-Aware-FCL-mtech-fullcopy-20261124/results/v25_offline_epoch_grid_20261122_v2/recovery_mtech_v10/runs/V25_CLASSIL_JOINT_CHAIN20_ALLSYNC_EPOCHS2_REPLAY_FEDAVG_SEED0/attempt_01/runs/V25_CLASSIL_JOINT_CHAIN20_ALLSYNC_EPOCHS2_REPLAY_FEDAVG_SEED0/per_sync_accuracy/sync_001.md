# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.83% on 6 observed labels.
Validation accuracy: 0.30666666666666664; seen labels never predicted: 2.
Server-accepted classes: 6; pending private classes: 0.
Accepted-only TEST accuracy: 0.35833333333333334.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **47.0%** (0: 60%, 1: 34%) |
| Client 1 | **46.0%** (1: 34%, 2: 58%) |
| Client 2 | **29.0%** (2: 58%, 3: 0%) |
| Client 3 | **31.5%** (3: 0%, 4: 63%) |
| Client 4 | **31.5%** (4: 63%, 5: 0%) |
