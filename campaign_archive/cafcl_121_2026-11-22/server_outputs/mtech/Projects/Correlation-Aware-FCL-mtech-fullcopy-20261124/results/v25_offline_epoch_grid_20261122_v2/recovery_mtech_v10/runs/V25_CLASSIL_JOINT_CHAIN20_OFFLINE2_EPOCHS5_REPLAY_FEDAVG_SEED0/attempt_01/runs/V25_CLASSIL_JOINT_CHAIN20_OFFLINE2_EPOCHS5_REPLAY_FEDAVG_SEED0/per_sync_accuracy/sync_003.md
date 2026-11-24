# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.25% on 16 observed labels.
Validation accuracy: 0.22; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.2325.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **31.5%** (0: 37%, 1: 26%) | **31.5%** (5: 31%, 6: 32%) | **0.5%** (10: 0%, 11: 1%) |
| Client 1 | **34.5%** (1: 26%, 2: 43%) | **48.0%** (6: 32%, 7: 64%) | **3.0%** (11: 1%, 12: 5%) |
| Client 2 | **36.5%** (2: 43%, 3: 30%) | **46.5%** (7: 64%, 8: 29%) | **3.0%** (12: 5%, 13: 1%) |
| Client 3 | **28.5%** (3: 30%, 4: 27%) | **18.0%** (8: 29%, 9: 7%) | **14.5%** (13: 1%, 14: 28%) |
| Client 4 | **29.0%** (4: 27%, 5: 31%) | **3.5%** (9: 7%, 10: 0%) | **19.5%** (14: 28%, 15: 11%) |
