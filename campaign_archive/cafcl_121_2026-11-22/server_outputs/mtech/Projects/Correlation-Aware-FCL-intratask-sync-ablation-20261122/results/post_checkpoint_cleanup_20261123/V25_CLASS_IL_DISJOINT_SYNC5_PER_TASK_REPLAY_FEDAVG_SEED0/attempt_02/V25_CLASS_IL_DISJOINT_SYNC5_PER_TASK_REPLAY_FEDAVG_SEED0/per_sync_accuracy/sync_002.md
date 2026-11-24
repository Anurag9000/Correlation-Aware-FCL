# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.90% on 10 observed labels.
Validation accuracy: 0.132; seen labels never predicted: 5.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.139.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **0.0%** (0: 0%, 1: 0%) |
| Client 1 | **5.5%** (2: 1%, 3: 10%) | **5.5%** (2: 1%, 3: 10%) |
| Client 2 | **19.5%** (4: 39%, 5: 0%) | **19.5%** (4: 39%, 5: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (6: 0%, 7: 0%) |
| Client 4 | **44.5%** (8: 89%, 9: 0%) | **44.5%** (8: 89%, 9: 0%) |
