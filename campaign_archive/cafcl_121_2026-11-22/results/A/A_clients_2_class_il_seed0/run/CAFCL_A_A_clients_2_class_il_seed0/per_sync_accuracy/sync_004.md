# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.88% on 16 observed labels.
Validation accuracy: 0.42625; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.41875.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **68.0%** (0: 90%, 1: 46%) | **50.5%** (4: 43%, 5: 58%) | **54.0%** (8: 60%, 9: 48%) | **23.5%** (12: 25%, 13: 22%) |
| Client 1 | **46.0%** (2: 49%, 3: 43%) | **56.0%** (6: 58%, 7: 54%) | **26.0%** (10: 17%, 11: 35%) | **11.0%** (14: 21%, 15: 1%) |
