# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.33% on 21 observed labels.
Validation accuracy: 0.22666666666666666; seen labels never predicted: 1.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.25333333333333335.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **63.5%** (0: 69%, 1: 58%) | **24.5%** (5: 12%, 6: 37%) | **4.0%** (10: 0%, 11: 8%) | **1.5%** (15: 3%, 16: 0%) |
| Client 1 | **46.0%** (1: 58%, 2: 34%) | **31.0%** (6: 37%, 7: 25%) | **33.0%** (11: 8%, 12: 58%) | **15.0%** (16: 0%, 17: 30%) |
| Client 2 | **33.0%** (2: 34%, 3: 32%) | **42.0%** (7: 25%, 8: 59%) | **30.0%** (12: 58%, 13: 2%) | **44.5%** (17: 30%, 18: 59%) |
| Client 3 | **34.0%** (3: 32%, 4: 36%) | **30.5%** (8: 59%, 9: 2%) | **1.0%** (13: 2%, 14: 0%) | **33.5%** (18: 59%, 19: 8%) |
| Client 4 | **24.0%** (4: 36%, 5: 12%) | **1.0%** (9: 2%, 10: 0%) | **1.5%** (14: 0%, 15: 3%) | **4.0%** (19: 8%, 20: 0%) |
