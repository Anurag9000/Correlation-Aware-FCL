# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.62% on 16 observed labels.
Validation accuracy: 0.175; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.18625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **50.5%** (0: 79%, 1: 22%) | **8.5%** (5: 6%, 6: 11%) | **14.5%** (10: 29%, 11: 0%) |
| Client 1 | **13.0%** (1: 22%, 2: 4%) | **5.5%** (6: 11%, 7: 0%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **31.0%** (2: 4%, 3: 58%) | **18.5%** (7: 0%, 8: 37%) | **0.0%** (12: 0%, 13: 0%) |
| Client 3 | **47.0%** (3: 58%, 4: 36%) | **24.0%** (8: 37%, 9: 11%) | **2.0%** (13: 0%, 14: 4%) |
| Client 4 | **21.0%** (4: 36%, 5: 6%) | **20.0%** (9: 11%, 10: 29%) | **2.5%** (14: 4%, 15: 1%) |
