# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.00% on 30 observed labels.
Validation accuracy: 0.052; seen labels never predicted: 15.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.06428571428571428.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **0.5%** (0: 0%, 1: 1%) | **1.5%** (10: 0%, 11: 3%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **30.5%** (2: 3%, 3: 58%) | **3.0%** (12: 5%, 13: 1%) | **0.0%** (22: 0%, 23: 0%) [PENDING upload] |
| Client 2 | **44.5%** (4: 70%, 5: 19%) | **0.0%** (14: 0%, 15: 0%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **7.0%** (6: 0%, 7: 14%) | **2.0%** (16: 3%, 17: 1%) | **0.0%** (26: 0%, 27: 0%) |
| Client 4 | **1.0%** (8: 2%, 9: 0%) | **0.0%** (18: 0%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) |
