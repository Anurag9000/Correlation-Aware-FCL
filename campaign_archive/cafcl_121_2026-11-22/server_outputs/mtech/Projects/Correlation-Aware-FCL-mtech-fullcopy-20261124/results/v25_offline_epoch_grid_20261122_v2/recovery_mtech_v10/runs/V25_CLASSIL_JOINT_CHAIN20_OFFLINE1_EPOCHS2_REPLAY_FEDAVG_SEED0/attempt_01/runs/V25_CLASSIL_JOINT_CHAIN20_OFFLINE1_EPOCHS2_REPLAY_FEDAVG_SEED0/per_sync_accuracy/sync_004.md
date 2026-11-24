# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.05% on 21 observed labels.
Validation accuracy: 0.21904761904761905; seen labels never predicted: 5.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.22047619047619046.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **71.5%** (0: 71%, 1: 72%) | **27.0%** (5: 29%, 6: 25%) | **8.0%** (10: 0%, 11: 16%) | **0.0%** (15: 0%, 16: 0%) |
| Client 1 | **47.5%** (1: 72%, 2: 23%) | **14.5%** (6: 25%, 7: 4%) | **27.5%** (11: 16%, 12: 39%) | **40.5%** (16: 0%, 17: 81%) |
| Client 2 | **15.0%** (2: 23%, 3: 7%) | **10.0%** (7: 4%, 8: 16%) | **19.5%** (12: 39%, 13: 0%) | **65.0%** (17: 81%, 18: 49%) |
| Client 3 | **18.0%** (3: 7%, 4: 29%) | **8.0%** (8: 16%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **25.5%** (18: 49%, 19: 2%) |
| Client 4 | **29.0%** (4: 29%, 5: 29%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **1.0%** (19: 2%, 20: 0%) |
