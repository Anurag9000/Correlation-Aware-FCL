# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.52% on 21 observed labels.
Validation accuracy: 0.1657142857142857; seen labels never predicted: 4.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.15523809523809523.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **65.0%** (0: 94%, 1: 36%) | **8.0%** (5: 10%, 6: 6%) | **16.5%** (10: 1%, 11: 32%) | **0.5%** (15: 0%, 16: 1%) |
| Client 1 | **19.5%** (1: 36%, 2: 3%) | **3.0%** (6: 6%, 7: 0%) | **16.5%** (11: 32%, 12: 1%) | **37.5%** (16: 1%, 17: 74%) |
| Client 2 | **3.0%** (2: 3%, 3: 3%) | **1.5%** (7: 0%, 8: 3%) | **0.5%** (12: 1%, 13: 0%) | **54.5%** (17: 74%, 18: 35%) |
| Client 3 | **11.5%** (3: 3%, 4: 20%) | **1.5%** (8: 3%, 9: 0%) | **1.0%** (13: 0%, 14: 2%) | **19.0%** (18: 35%, 19: 3%) |
| Client 4 | **15.0%** (4: 20%, 5: 10%) | **0.5%** (9: 0%, 10: 1%) | **1.0%** (14: 2%, 15: 0%) | **2.5%** (19: 3%, 20: 2%) |
