# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.10% on 21 observed labels.
Validation accuracy: 0.1419047619047619; seen labels never predicted: 9.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.18777777777777777.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **11.0%** (0: 0%, 1: 22%) [PENDING upload] | **21.5%** (5: 43%, 6: 0%) | **8.0%** (10: 16%, 11: 0%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] |
| Client 1 | **27.5%** (1: 22%, 2: 33%) | **27.5%** (6: 0%, 7: 55%) | **0.0%** (11: 0%, 12: 0%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **17.5%** (2: 33%, 3: 2%) | **30.0%** (7: 55%, 8: 5%) | **0.0%** (12: 0%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **36.0%** (3: 2%, 4: 70%) | **8.0%** (8: 5%, 9: 11%) | **2.5%** (13: 0%, 14: 5%) | **25.5%** (18: 0%, 19: 51%) |
| Client 4 | **56.5%** (4: 70%, 5: 43%) | **13.5%** (9: 11%, 10: 16%) | **2.5%** (14: 5%, 15: 0%) | **38.0%** (19: 51%, 20: 25%) |
