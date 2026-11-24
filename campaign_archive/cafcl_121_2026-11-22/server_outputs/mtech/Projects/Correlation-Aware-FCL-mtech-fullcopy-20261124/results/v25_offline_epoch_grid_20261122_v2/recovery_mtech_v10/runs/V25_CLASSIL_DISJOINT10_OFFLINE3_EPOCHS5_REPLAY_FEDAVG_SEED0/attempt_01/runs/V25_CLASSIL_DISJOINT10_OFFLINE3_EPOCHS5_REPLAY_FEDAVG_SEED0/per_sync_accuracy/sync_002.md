# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 5.10% on 20 observed labels.
Validation accuracy: 0.051; seen labels never predicted: 15.
Server-accepted classes: 10; pending private classes: 10.
Accepted-only TEST accuracy: 0.1.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **1.0%** (0: 0%, 1: 2%) [PENDING upload] | **0.0%** (10: 0%, 11: 0%) [PENDING upload] |
| Client 1 | **0.0%** (2: 0%, 3: 0%) | **0.0%** (12: 0%, 13: 0%) [PENDING upload] |
| Client 2 | **0.0%** (4: 0%, 5: 0%) | **0.0%** (14: 0%, 15: 0%) |
| Client 3 | **0.0%** (6: 0%, 7: 0%) [PENDING upload] | **0.0%** (16: 0%, 17: 0%) [PENDING upload] |
| Client 4 | **0.0%** (8: 0%, 9: 0%) | **50.0%** (18: 100%, 19: 0%) |
