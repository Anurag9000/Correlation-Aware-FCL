# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.74% on 31 observed labels.
Validation accuracy: 0.14; seen labels never predicted: 14.
Server-accepted classes: 31; pending private classes: 0.
Accepted-only TEST accuracy: 0.12741935483870967.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **45.0%** (0: 54%, 1: 36%) | **11.0%** (5: 0%, 6: 22%) | **3.0%** (10: 0%, 11: 6%) | **0.0%** (15: 0%, 16: 0%) | **28.5%** (20: 43%, 21: 14%) | **18.5%** (25: 0%, 26: 37%) |
| Client 1 | **29.0%** (1: 36%, 2: 22%) | **11.0%** (6: 22%, 7: 0%) | **3.0%** (11: 6%, 12: 0%) | **32.5%** (16: 0%, 17: 65%) | **7.0%** (21: 14%, 22: 0%) | **18.5%** (26: 37%, 27: 0%) |
| Client 2 | **16.0%** (2: 22%, 3: 10%) | **18.5%** (7: 0%, 8: 37%) | **0.0%** (12: 0%, 13: 0%) | **42.5%** (17: 65%, 18: 20%) | **0.0%** (22: 0%, 23: 0%) | **0.0%** (27: 0%, 28: 0%) |
| Client 3 | **11.5%** (3: 10%, 4: 13%) | **18.5%** (8: 37%, 9: 0%) | **2.5%** (13: 0%, 14: 5%) | **10.0%** (18: 20%, 19: 0%) | **0.0%** (23: 0%, 24: 0%) | **5.5%** (28: 0%, 29: 11%) |
| Client 4 | **6.5%** (4: 13%, 5: 0%) | **0.0%** (9: 0%, 10: 0%) | **2.5%** (14: 5%, 15: 0%) | **21.5%** (19: 0%, 20: 43%) | **0.0%** (24: 0%, 25: 0%) | **5.5%** (29: 11%, 30: 0%) |
