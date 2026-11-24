# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 13.29% on 21 observed labels.
Validation accuracy: 0.12571428571428572; seen labels never predicted: 8.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.155.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **33.0%** (0: 0%, 1: 66%) [PENDING upload] | **10.5%** (5: 16%, 6: 5%) | **1.0%** (10: 2%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] |
| Client 1 | **37.5%** (1: 66%, 2: 9%) | **24.5%** (6: 5%, 7: 44%) | **6.0%** (11: 0%, 12: 12%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **4.5%** (2: 9%, 3: 0%) | **23.0%** (7: 44%, 8: 2%) | **6.0%** (12: 12%, 13: 0%) | **1.0%** (17: 0%, 18: 2%) [PENDING upload] |
| Client 3 | **44.0%** (3: 0%, 4: 88%) | **1.0%** (8: 2%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **16.5%** (18: 2%, 19: 31%) |
| Client 4 | **52.0%** (4: 88%, 5: 16%) | **1.0%** (9: 0%, 10: 2%) | **0.0%** (14: 0%, 15: 0%) | **16.5%** (19: 31%, 20: 2%) |
