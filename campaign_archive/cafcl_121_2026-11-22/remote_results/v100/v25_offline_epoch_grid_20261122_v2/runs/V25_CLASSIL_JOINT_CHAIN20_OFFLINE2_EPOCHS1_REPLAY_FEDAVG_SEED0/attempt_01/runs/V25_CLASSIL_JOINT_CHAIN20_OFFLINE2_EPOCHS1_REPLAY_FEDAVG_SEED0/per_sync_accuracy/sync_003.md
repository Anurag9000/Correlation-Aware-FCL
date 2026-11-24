# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.81% on 16 observed labels.
Validation accuracy: 0.19375; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.178125.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **46.5%** (0: 86%, 1: 7%) | **19.5%** (5: 38%, 6: 1%) | **21.5%** (10: 0%, 11: 43%) |
| Client 1 | **5.5%** (1: 7%, 2: 4%) | **0.5%** (6: 1%, 7: 0%) | **21.5%** (11: 43%, 12: 0%) |
| Client 2 | **2.0%** (2: 4%, 3: 0%) | **4.5%** (7: 0%, 8: 9%) | **4.5%** (12: 0%, 13: 9%) |
| Client 3 | **18.0%** (3: 0%, 4: 36%) | **4.5%** (8: 9%, 9: 0%) | **23.0%** (13: 9%, 14: 37%) |
| Client 4 | **37.0%** (4: 36%, 5: 38%) | **0.0%** (9: 0%, 10: 0%) | **26.0%** (14: 37%, 15: 15%) |
