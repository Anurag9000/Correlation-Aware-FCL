# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.23% on 31 observed labels.
Validation accuracy: 0.1406451612903226; seen labels never predicted: 11.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.1520689655172414.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **28.0%** (0: 0%, 1: 56%) | **25.0%** (5: 41%, 6: 9%) | **1.0%** (10: 2%, 11: 0%) | **1.0%** (15: 2%, 16: 0%) | **6.0%** (20: 12%, 21: 0%) | **5.5%** (25: 11%, 26: 0%) |
| Client 1 | **36.5%** (1: 56%, 2: 17%) | **28.5%** (6: 9%, 7: 48%) | **30.5%** (11: 0%, 12: 61%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] |
| Client 2 | **11.5%** (2: 17%, 3: 6%) | **34.0%** (7: 48%, 8: 20%) | **33.5%** (12: 61%, 13: 6%) | **16.0%** (17: 0%, 18: 32%) | **0.5%** (22: 0%, 23: 1%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] |
| Client 3 | **45.0%** (3: 6%, 4: 84%) | **15.0%** (8: 20%, 9: 10%) | **3.0%** (13: 6%, 14: 0%) | **26.0%** (18: 32%, 19: 20%) | **2.0%** (23: 1%, 24: 3%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
| Client 4 | **62.5%** (4: 84%, 5: 41%) | **6.0%** (9: 10%, 10: 2%) | **1.0%** (14: 0%, 15: 2%) | **16.0%** (19: 20%, 20: 12%) | **7.0%** (24: 3%, 25: 11%) | **0.0%** (29: 0%, 30: 0%) |
