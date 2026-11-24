# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.10% on 31 observed labels.
Validation accuracy: 0.10451612903225807; seen labels never predicted: 15.
Server-accepted classes: 31; pending private classes: 0.
Accepted-only TEST accuracy: 0.11096774193548387.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **31.5%** (0: 52%, 1: 11%) | **10.0%** (5: 0%, 6: 20%) | **1.0%** (10: 0%, 11: 2%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) | **6.0%** (25: 0%, 26: 12%) |
| Client 1 | **24.5%** (1: 11%, 2: 38%) | **10.0%** (6: 20%, 7: 0%) | **3.5%** (11: 2%, 12: 5%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) | **9.0%** (26: 12%, 27: 6%) |
| Client 2 | **19.0%** (2: 38%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **2.5%** (12: 5%, 13: 0%) | **20.5%** (17: 0%, 18: 41%) | **34.5%** (22: 0%, 23: 69%) | **3.0%** (27: 6%, 28: 0%) |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **0.0%** (8: 0%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **20.5%** (18: 41%, 19: 0%) | **34.5%** (23: 69%, 24: 0%) | **44.0%** (28: 0%, 29: 88%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **0.0%** (24: 0%, 25: 0%) | **44.0%** (29: 88%, 30: 0%) |
