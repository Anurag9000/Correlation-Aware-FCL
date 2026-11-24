# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.05% on 21 observed labels.
Validation accuracy: 0.15619047619047619; seen labels never predicted: 5.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.1685.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **52.0%** (0: 98%, 1: 6%) | **18.0%** (5: 21%, 6: 15%) | **20.5%** (10: 0%, 11: 41%) | **16.5%** (15: 33%, 16: 0%) |
| Client 1 | **3.0%** (1: 6%, 2: 0%) | **7.5%** (6: 15%, 7: 0%) | **42.5%** (11: 41%, 12: 44%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **9.0%** (7: 0%, 8: 18%) | **22.0%** (12: 44%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **12.0%** (3: 0%, 4: 24%) | **9.0%** (8: 18%, 9: 0%) | **15.0%** (13: 0%, 14: 30%) | **3.0%** (18: 0%, 19: 6%) |
| Client 4 | **22.5%** (4: 24%, 5: 21%) | **0.0%** (9: 0%, 10: 0%) | **31.5%** (14: 30%, 15: 33%) | **3.5%** (19: 6%, 20: 1%) |
