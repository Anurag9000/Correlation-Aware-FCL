# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.50% on 10 observed labels.
Validation accuracy: 0.176; seen labels never predicted: 4.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.175.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **10.0%** (0: 20%, 1: 0%) | **10.0%** (0: 20%, 1: 0%) |
| Client 1 | **14.0%** (2: 0%, 3: 28%) | **14.0%** (2: 0%, 3: 28%) |
| Client 2 | **40.5%** (4: 79%, 5: 2%) | **40.5%** (4: 79%, 5: 2%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) | **0.0%** (6: 0%, 7: 0%) |
| Client 4 | **23.0%** (8: 36%, 9: 10%) | **23.0%** (8: 36%, 9: 10%) |
