# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.42% on 26 observed labels.
Validation accuracy: 0.21307692307692308; seen labels never predicted: 3.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.2228.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **47.0%** (0: 44%, 1: 50%) | **22.0%** (5: 12%, 6: 32%) | **3.0%** (10: 0%, 11: 6%) | **2.5%** (15: 5%, 16: 0%) | **40.0%** (20: 40%, 21: 40%) |
| Client 1 | **38.0%** (1: 50%, 2: 26%) | **16.0%** (6: 32%, 7: 0%) | **33.0%** (11: 6%, 12: 60%) | **31.5%** (16: 0%, 17: 63%) | **20.0%** (21: 40%, 22: 0%) |
| Client 2 | **24.0%** (2: 26%, 3: 22%) | **7.0%** (7: 0%, 8: 14%) | **30.0%** (12: 60%, 13: 0%) | **56.5%** (17: 63%, 18: 50%) | **29.5%** (22: 0%, 23: 59%) |
| Client 3 | **23.5%** (3: 22%, 4: 25%) | **7.0%** (8: 14%, 9: 0%) | **3.5%** (13: 0%, 14: 7%) | **26.0%** (18: 50%, 19: 2%) | **29.5%** (23: 59%, 24: 0%) |
| Client 4 | **18.5%** (4: 25%, 5: 12%) | **0.0%** (9: 0%, 10: 0%) | **6.0%** (14: 7%, 15: 5%) | **21.0%** (19: 2%, 20: 40%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
