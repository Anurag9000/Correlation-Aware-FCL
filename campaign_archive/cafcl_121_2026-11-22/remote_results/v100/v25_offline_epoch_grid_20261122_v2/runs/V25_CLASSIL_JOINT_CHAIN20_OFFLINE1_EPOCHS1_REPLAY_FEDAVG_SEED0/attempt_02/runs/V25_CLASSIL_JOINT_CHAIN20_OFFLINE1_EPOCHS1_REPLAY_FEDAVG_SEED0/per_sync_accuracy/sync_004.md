# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.90% on 21 observed labels.
Validation accuracy: 0.18857142857142858; seen labels never predicted: 4.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.17904761904761904.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **64.5%** (0: 92%, 1: 37%) | **6.5%** (5: 11%, 6: 2%) | **11.5%** (10: 0%, 11: 23%) | **0.0%** (15: 0%, 16: 0%) |
| Client 1 | **24.0%** (1: 37%, 2: 11%) | **1.0%** (6: 2%, 7: 0%) | **25.0%** (11: 23%, 12: 27%) | **35.0%** (16: 0%, 17: 70%) |
| Client 2 | **7.0%** (2: 11%, 3: 3%) | **5.5%** (7: 0%, 8: 11%) | **13.5%** (12: 27%, 13: 0%) | **54.0%** (17: 70%, 18: 38%) |
| Client 3 | **14.0%** (3: 3%, 4: 25%) | **5.5%** (8: 11%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **19.0%** (18: 38%, 19: 0%) |
| Client 4 | **18.0%** (4: 25%, 5: 11%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **13.0%** (19: 0%, 20: 26%) |
