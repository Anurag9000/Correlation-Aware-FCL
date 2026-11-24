# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.52% on 21 observed labels.
Validation accuracy: 0.14285714285714285; seen labels never predicted: 6.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.15523809523809523.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **57.0%** (0: 94%, 1: 20%) | **9.5%** (5: 11%, 6: 8%) | **19.5%** (10: 1%, 11: 38%) | **0.0%** (15: 0%, 16: 0%) |
| Client 1 | **11.5%** (1: 20%, 2: 3%) | **4.0%** (6: 8%, 7: 0%) | **23.5%** (11: 38%, 12: 9%) | **28.0%** (16: 0%, 17: 56%) |
| Client 2 | **1.5%** (2: 3%, 3: 0%) | **2.5%** (7: 0%, 8: 5%) | **5.0%** (12: 9%, 13: 1%) | **47.0%** (17: 56%, 18: 38%) |
| Client 3 | **15.5%** (3: 0%, 4: 31%) | **2.5%** (8: 5%, 9: 0%) | **0.5%** (13: 1%, 14: 0%) | **20.0%** (18: 38%, 19: 2%) |
| Client 4 | **21.0%** (4: 31%, 5: 11%) | **0.5%** (9: 0%, 10: 1%) | **0.0%** (14: 0%, 15: 0%) | **5.5%** (19: 2%, 20: 9%) |
