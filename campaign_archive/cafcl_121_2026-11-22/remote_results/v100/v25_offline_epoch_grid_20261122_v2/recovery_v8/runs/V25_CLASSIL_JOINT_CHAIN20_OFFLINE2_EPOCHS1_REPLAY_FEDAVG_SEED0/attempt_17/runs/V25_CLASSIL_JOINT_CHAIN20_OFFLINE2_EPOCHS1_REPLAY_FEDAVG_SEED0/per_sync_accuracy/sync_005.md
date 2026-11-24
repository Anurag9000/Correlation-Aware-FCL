# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.35% on 26 observed labels.
Validation accuracy: 0.1123076923076923; seen labels never predicted: 9.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.12346153846153846.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **66.0%** (0: 98%, 1: 34%) | **22.5%** (5: 5%, 6: 40%) | **0.5%** (10: 1%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **1.0%** (20: 2%, 21: 0%) |
| Client 1 | **17.5%** (1: 34%, 2: 1%) | **21.0%** (6: 40%, 7: 2%) | **34.0%** (11: 0%, 12: 68%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **1.5%** (2: 1%, 3: 2%) | **3.5%** (7: 2%, 8: 5%) | **35.5%** (12: 68%, 13: 3%) | **0.0%** (17: 0%, 18: 0%) | **0.5%** (22: 0%, 23: 1%) |
| Client 3 | **22.0%** (3: 2%, 4: 42%) | **5.5%** (8: 5%, 9: 6%) | **2.5%** (13: 3%, 14: 2%) | **0.0%** (18: 0%, 19: 0%) | **5.0%** (23: 1%, 24: 9%) |
| Client 4 | **23.5%** (4: 42%, 5: 5%) | **3.5%** (9: 6%, 10: 1%) | **1.0%** (14: 2%, 15: 0%) | **1.0%** (19: 0%, 20: 2%) | **4.5%** (24: 9%, 25: 0%) |
