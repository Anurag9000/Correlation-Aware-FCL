# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.05% on 21 observed labels.
Validation accuracy: 0.15047619047619049; seen labels never predicted: 5.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.15047619047619049.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **62.0%** (0: 98%, 1: 26%) | **8.5%** (5: 17%, 6: 0%) | **4.0%** (10: 3%, 11: 5%) | **0.0%** (15: 0%, 16: 0%) |
| Client 1 | **16.0%** (1: 26%, 2: 6%) | **0.5%** (6: 0%, 7: 1%) | **8.0%** (11: 5%, 12: 11%) | **30.5%** (16: 0%, 17: 61%) |
| Client 2 | **3.0%** (2: 6%, 3: 0%) | **1.5%** (7: 1%, 8: 2%) | **6.0%** (12: 11%, 13: 1%) | **47.5%** (17: 61%, 18: 34%) |
| Client 3 | **5.5%** (3: 0%, 4: 11%) | **6.5%** (8: 2%, 9: 11%) | **0.5%** (13: 1%, 14: 0%) | **27.5%** (18: 34%, 19: 21%) |
| Client 4 | **14.0%** (4: 11%, 5: 17%) | **7.0%** (9: 11%, 10: 3%) | **0.0%** (14: 0%, 15: 0%) | **14.5%** (19: 21%, 20: 8%) |
