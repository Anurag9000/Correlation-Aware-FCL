# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.18% on 11 observed labels.
Validation accuracy: 0.18545454545454546; seen labels never predicted: 5.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.23444444444444446.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **30.5%** (0: 0%, 1: 61%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) [PENDING upload] |
| Client 1 | **58.0%** (1: 61%, 2: 55%) | **0.5%** (6: 0%, 7: 1%) [PENDING upload] |
| Client 2 | **27.5%** (2: 55%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) |
| Client 3 | **32.5%** (3: 0%, 4: 65%) | **14.5%** (8: 0%, 9: 29%) |
| Client 4 | **32.5%** (4: 65%, 5: 0%) | **14.5%** (9: 29%, 10: 0%) |
