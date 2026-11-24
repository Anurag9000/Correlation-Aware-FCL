# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 12.88% on 26 observed labels.
Validation accuracy: 0.1076923076923077; seen labels never predicted: 10.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.12884615384615383.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **60.5%** (0: 39%, 1: 82%) | **20.0%** (5: 6%, 6: 34%) | **0.0%** (10: 0%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) | **5.5%** (20: 11%, 21: 0%) |
| Client 1 | **41.5%** (1: 82%, 2: 1%) | **17.0%** (6: 34%, 7: 0%) | **24.0%** (11: 0%, 12: 48%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **0.5%** (2: 1%, 3: 0%) | **2.5%** (7: 0%, 8: 5%) | **25.0%** (12: 48%, 13: 2%) | **0.0%** (17: 0%, 18: 0%) | **0.0%** (22: 0%, 23: 0%) |
| Client 3 | **26.0%** (3: 0%, 4: 52%) | **6.0%** (8: 5%, 9: 7%) | **7.5%** (13: 2%, 14: 13%) | **0.0%** (18: 0%, 19: 0%) | **17.5%** (23: 0%, 24: 35%) |
| Client 4 | **29.0%** (4: 52%, 5: 6%) | **3.5%** (9: 7%, 10: 0%) | **6.5%** (14: 13%, 15: 0%) | **5.5%** (19: 0%, 20: 11%) | **17.5%** (24: 35%, 25: 0%) |
