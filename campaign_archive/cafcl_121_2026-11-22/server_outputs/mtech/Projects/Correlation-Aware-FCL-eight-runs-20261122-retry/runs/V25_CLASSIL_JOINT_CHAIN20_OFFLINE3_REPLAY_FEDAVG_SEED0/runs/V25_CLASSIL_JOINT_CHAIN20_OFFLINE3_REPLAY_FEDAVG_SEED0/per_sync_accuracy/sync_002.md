# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.09% on 11 observed labels.
Validation accuracy: 0.16363636363636364; seen labels never predicted: 4.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.2222222222222222.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **10.5%** (0: 0%, 1: 21%) [PENDING upload] | **14.0%** (5: 28%, 6: 0%) [PENDING upload] |
| Client 1 | **33.0%** (1: 21%, 2: 45%) | **0.5%** (6: 0%, 7: 1%) [PENDING upload] |
| Client 2 | **22.5%** (2: 45%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) |
| Client 3 | **18.5%** (3: 0%, 4: 37%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **32.5%** (4: 37%, 5: 28%) | **33.5%** (9: 0%, 10: 67%) |
