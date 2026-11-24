# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.77% on 26 observed labels.
Validation accuracy: 0.19076923076923077; seen labels never predicted: 6.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.1976923076923077.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **44.5%** (0: 32%, 1: 57%) | **33.0%** (5: 37%, 6: 29%) | **1.5%** (10: 0%, 11: 3%) | **10.5%** (15: 21%, 16: 0%) | **5.0%** (20: 0%, 21: 10%) |
| Client 1 | **51.0%** (1: 57%, 2: 45%) | **17.5%** (6: 29%, 7: 6%) | **40.0%** (11: 3%, 12: 77%) | **0.0%** (16: 0%, 17: 0%) | **5.0%** (21: 10%, 22: 0%) |
| Client 2 | **25.0%** (2: 45%, 3: 5%) | **19.5%** (7: 6%, 8: 33%) | **44.0%** (12: 77%, 13: 11%) | **7.0%** (17: 0%, 18: 14%) | **2.0%** (22: 0%, 23: 4%) |
| Client 3 | **15.0%** (3: 5%, 4: 25%) | **16.5%** (8: 33%, 9: 0%) | **9.0%** (13: 11%, 14: 7%) | **19.0%** (18: 14%, 19: 24%) | **39.0%** (23: 4%, 24: 74%) |
| Client 4 | **31.0%** (4: 25%, 5: 37%) | **0.0%** (9: 0%, 10: 0%) | **14.0%** (14: 7%, 15: 21%) | **12.0%** (19: 24%, 20: 0%) | **37.0%** (24: 74%, 25: 0%) |
