# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 9.20% on 20 observed labels.
Validation accuracy: 0.098; seen labels never predicted: 9.
Server-accepted classes: 14; pending private classes: 6.
Accepted-only TEST accuracy: 0.13142857142857142.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **11.0%** (0: 22%, 1: 0%) | **0.0%** (10: 0%, 11: 0%) [PENDING upload] |
| Client 1 | **15.5%** (2: 28%, 3: 3%) | **47.5%** (12: 94%, 13: 1%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] |
| Client 3 | **0.5%** (6: 1%, 7: 0%) | **0.5%** (16: 1%, 17: 0%) |
| Client 4 | **3.5%** (8: 4%, 9: 3%) | **13.5%** (18: 24%, 19: 3%) |
