# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 6.56% on 16 observed labels.
Validation accuracy: 0.0625; seen labels never predicted: 12.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.075.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **48.5%** (0: 0%, 1: 97%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) | **2.5%** (10: 5%, 11: 0%) |
| Client 1 | **48.5%** (1: 97%, 2: 0%) | **0.0%** (6: 0%, 7: 0%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (7: 0%, 8: 0%) | **0.0%** (12: 0%, 13: 0%) |
| Client 3 | **0.0%** (3: 0%, 4: 0%) | **1.5%** (8: 0%, 9: 3%) | **0.0%** (13: 0%, 14: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) | **4.0%** (9: 3%, 10: 5%) | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
