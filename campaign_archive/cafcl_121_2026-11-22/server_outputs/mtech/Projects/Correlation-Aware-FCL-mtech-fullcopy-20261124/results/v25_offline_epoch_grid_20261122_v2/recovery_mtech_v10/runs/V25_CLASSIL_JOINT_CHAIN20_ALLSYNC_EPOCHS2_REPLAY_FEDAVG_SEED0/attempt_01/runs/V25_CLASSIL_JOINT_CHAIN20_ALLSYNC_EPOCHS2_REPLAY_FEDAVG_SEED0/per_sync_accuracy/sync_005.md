# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.92% on 26 observed labels.
Validation accuracy: 0.21307692307692308; seen labels never predicted: 4.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.22923076923076924.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **35.5%** (0: 46%, 1: 25%) | **35.5%** (5: 38%, 6: 33%) | **0.0%** (10: 0%, 11: 0%) | **13.5%** (15: 27%, 16: 0%) | **53.5%** (20: 56%, 21: 51%) |
| Client 1 | **23.5%** (1: 25%, 2: 22%) | **18.0%** (6: 33%, 7: 3%) | **4.5%** (11: 0%, 12: 9%) | **36.5%** (16: 0%, 17: 73%) | **25.5%** (21: 51%, 22: 0%) |
| Client 2 | **12.5%** (2: 22%, 3: 3%) | **11.5%** (7: 3%, 8: 20%) | **6.5%** (12: 9%, 13: 4%) | **66.0%** (17: 73%, 18: 59%) | **36.5%** (22: 0%, 23: 73%) |
| Client 3 | **15.5%** (3: 3%, 4: 28%) | **10.0%** (8: 20%, 9: 0%) | **6.5%** (13: 4%, 14: 9%) | **33.0%** (18: 59%, 19: 7%) | **41.5%** (23: 73%, 24: 10%) |
| Client 4 | **33.0%** (4: 28%, 5: 38%) | **0.0%** (9: 0%, 10: 0%) | **18.0%** (14: 9%, 15: 27%) | **31.5%** (19: 7%, 20: 56%) | **5.0%** (24: 10%, 25: 0%) |
