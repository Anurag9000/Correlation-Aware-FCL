# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.56% on 36 observed labels.
Validation accuracy: 0.23555555555555555; seen labels never predicted: 5.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.265.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **46.0%** (0: 48%, 1: 44%) | **30.5%** (5: 45%, 6: 16%) | **3.0%** (10: 0%, 11: 6%) | **7.0%** (15: 14%, 16: 0%) | **30.5%** (20: 33%, 21: 28%) | **3.5%** (25: 7%, 26: 0%) | **60.5%** (30: 95%, 31: 26%) |
| Client 1 | **34.0%** (1: 44%, 2: 24%) | **14.5%** (6: 16%, 7: 13%) | **19.0%** (11: 6%, 12: 32%) | **37.5%** (16: 0%, 17: 75%) | **15.0%** (21: 28%, 22: 2%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] | **13.0%** (31: 26%, 32: 0%) [PENDING upload] |
| Client 2 | **33.5%** (2: 24%, 3: 43%) | **15.5%** (7: 13%, 8: 18%) | **33.0%** (12: 32%, 13: 34%) | **56.5%** (17: 75%, 18: 38%) | **13.0%** (22: 2%, 23: 24%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 3 | **40.0%** (3: 43%, 4: 37%) | **24.5%** (8: 18%, 9: 31%) | **20.0%** (13: 34%, 14: 6%) | **26.5%** (18: 38%, 19: 15%) | **41.5%** (23: 24%, 24: 59%) | **1.5%** (28: 0%, 29: 3%) [PENDING upload] | **10.5%** (33: 0%, 34: 21%) [PENDING upload] |
| Client 4 | **41.0%** (4: 37%, 5: 45%) | **15.5%** (9: 31%, 10: 0%) | **10.0%** (14: 6%, 15: 14%) | **24.0%** (19: 15%, 20: 33%) | **33.0%** (24: 59%, 25: 7%) | **49.0%** (29: 3%, 30: 95%) | **16.0%** (34: 21%, 35: 11%) |
