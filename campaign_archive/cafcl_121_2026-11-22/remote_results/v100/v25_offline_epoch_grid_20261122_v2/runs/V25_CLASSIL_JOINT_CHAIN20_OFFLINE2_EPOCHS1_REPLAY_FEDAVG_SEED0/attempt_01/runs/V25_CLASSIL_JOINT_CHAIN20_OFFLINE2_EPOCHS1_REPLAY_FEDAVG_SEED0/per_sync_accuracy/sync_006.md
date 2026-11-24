# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 9.06% on 31 observed labels.
Validation accuracy: 0.08193548387096775; seen labels never predicted: 13.
Server-accepted classes: 31; pending private classes: 0.
Accepted-only TEST accuracy: 0.09064516129032259.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **10.0%** (0: 19%, 1: 1%) | **10.0%** (5: 0%, 6: 20%) | **6.5%** (10: 0%, 11: 13%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) | **3.5%** (25: 1%, 26: 6%) |
| Client 1 | **15.5%** (1: 1%, 2: 30%) | **10.0%** (6: 20%, 7: 0%) | **11.0%** (11: 13%, 12: 9%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) | **3.0%** (26: 6%, 27: 0%) |
| Client 2 | **15.0%** (2: 30%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **4.5%** (12: 9%, 13: 0%) | **7.5%** (17: 0%, 18: 15%) | **35.5%** (22: 0%, 23: 71%) | **0.0%** (27: 0%, 28: 0%) |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **0.0%** (8: 0%, 9: 0%) | **0.0%** (13: 0%, 14: 0%) | **7.5%** (18: 15%, 19: 0%) | **35.5%** (23: 71%, 24: 0%) | **46.0%** (28: 0%, 29: 92%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **0.5%** (24: 0%, 25: 1%) | **48.0%** (29: 92%, 30: 4%) |
