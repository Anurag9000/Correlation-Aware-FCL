# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 8.35% on 31 observed labels.
Validation accuracy: 0.08838709677419355; seen labels never predicted: 15.
Server-accepted classes: 31; pending private classes: 0.
Accepted-only TEST accuracy: 0.08354838709677419.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **4.0%** (0: 0%, 1: 8%) | **13.0%** (5: 1%, 6: 25%) | **4.5%** (10: 0%, 11: 9%) | **0.0%** (15: 0%, 16: 0%) | **3.0%** (20: 0%, 21: 6%) | **6.5%** (25: 0%, 26: 13%) |
| Client 1 | **9.0%** (1: 8%, 2: 10%) | **12.5%** (6: 25%, 7: 0%) | **17.5%** (11: 9%, 12: 26%) | **0.0%** (16: 0%, 17: 0%) | **3.0%** (21: 6%, 22: 0%) | **8.0%** (26: 13%, 27: 3%) |
| Client 2 | **5.0%** (2: 10%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **13.5%** (12: 26%, 13: 1%) | **0.0%** (17: 0%, 18: 0%) | **36.5%** (22: 0%, 23: 73%) | **2.5%** (27: 3%, 28: 2%) |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **0.0%** (8: 0%, 9: 0%) | **0.5%** (13: 1%, 14: 0%) | **0.0%** (18: 0%, 19: 0%) | **36.5%** (23: 73%, 24: 0%) | **42.0%** (28: 2%, 29: 82%) |
| Client 4 | **0.5%** (4: 0%, 5: 1%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **0.0%** (24: 0%, 25: 0%) | **41.0%** (29: 82%, 30: 0%) |
