# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.46% on 26 observed labels.
Validation accuracy: 0.17; seen labels never predicted: 7.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.1712.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **64.0%** (0: 55%, 1: 73%) | **18.5%** (5: 3%, 6: 34%) | **2.5%** (10: 0%, 11: 5%) | **0.0%** (15: 0%, 16: 0%) | **48.5%** (20: 71%, 21: 26%) |
| Client 1 | **44.5%** (1: 73%, 2: 16%) | **17.0%** (6: 34%, 7: 0%) | **42.0%** (11: 5%, 12: 79%) | **5.5%** (16: 0%, 17: 11%) | **14.5%** (21: 26%, 22: 3%) |
| Client 2 | **8.0%** (2: 16%, 3: 0%) | **3.5%** (7: 0%, 8: 7%) | **39.5%** (12: 79%, 13: 0%) | **10.0%** (17: 11%, 18: 9%) | **5.5%** (22: 3%, 23: 8%) |
| Client 3 | **11.0%** (3: 0%, 4: 22%) | **3.5%** (8: 7%, 9: 0%) | **3.0%** (13: 0%, 14: 6%) | **4.5%** (18: 9%, 19: 0%) | **4.0%** (23: 8%, 24: 0%) |
| Client 4 | **12.5%** (4: 22%, 5: 3%) | **0.0%** (9: 0%, 10: 0%) | **3.0%** (14: 6%, 15: 0%) | **35.5%** (19: 0%, 20: 71%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
