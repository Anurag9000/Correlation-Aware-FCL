# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.70% on 10 observed labels.
Validation accuracy: 0.236; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.257.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **28.0%** (0: 47%, 1: 9%) | **28.0%** (0: 47%, 1: 9%) | **28.0%** (0: 47%, 1: 9%) |
| Client 1 | **28.5%** (2: 17%, 3: 40%) | **28.5%** (2: 17%, 3: 40%) | **28.5%** (2: 17%, 3: 40%) |
| Client 2 | **20.0%** (4: 40%, 5: 0%) | **20.0%** (4: 40%, 5: 0%) | **20.0%** (4: 40%, 5: 0%) |
| Client 3 | **8.0%** (6: 12%, 7: 4%) | **8.0%** (6: 12%, 7: 4%) | **8.0%** (6: 12%, 7: 4%) |
| Client 4 | **44.0%** (8: 75%, 9: 13%) | **44.0%** (8: 75%, 9: 13%) | **44.0%** (8: 75%, 9: 13%) |
