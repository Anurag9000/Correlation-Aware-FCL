# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.94% on 16 observed labels.
Validation accuracy: 0.32375; seen labels never predicted: 1.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.43142857142857144.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **47.5%** (0: 0%, 1: 95%) [PENDING upload] | **46.0%** (5: 0%, 6: 92%) | **42.0%** (10: 0%, 11: 84%) |
| Client 1 | **51.0%** (1: 95%, 2: 7%) | **66.0%** (6: 92%, 7: 40%) | **42.0%** (11: 84%, 12: 0%) |
| Client 2 | **4.0%** (2: 7%, 3: 1%) | **29.5%** (7: 40%, 8: 19%) | **3.0%** (12: 0%, 13: 6%) |
| Client 3 | **32.0%** (3: 1%, 4: 63%) | **53.5%** (8: 19%, 9: 88%) | **3.0%** (13: 6%, 14: 0%) |
| Client 4 | **31.5%** (4: 63%, 5: 0%) | **44.0%** (9: 88%, 10: 0%) | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
