# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.86% on 21 observed labels.
Validation accuracy: 0.11904761904761904; seen labels never predicted: 7.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.1245.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **59.0%** (0: 99%, 1: 19%) | **6.0%** (5: 8%, 6: 4%) | **23.5%** (10: 0%, 11: 47%) | **1.5%** (15: 3%, 16: 0%) |
| Client 1 | **9.5%** (1: 19%, 2: 0%) | **2.0%** (6: 4%, 7: 0%) | **27.5%** (11: 47%, 12: 8%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.5%** (7: 0%, 8: 1%) | **4.0%** (12: 8%, 13: 0%) | **0.0%** (17: 0%, 18: 0%) [PENDING upload] |
| Client 3 | **19.0%** (3: 0%, 4: 38%) | **0.5%** (8: 1%, 9: 0%) | **8.5%** (13: 0%, 14: 17%) | **2.5%** (18: 0%, 19: 5%) |
| Client 4 | **23.0%** (4: 38%, 5: 8%) | **0.0%** (9: 0%, 10: 0%) | **10.0%** (14: 17%, 15: 3%) | **2.5%** (19: 5%, 20: 0%) |
