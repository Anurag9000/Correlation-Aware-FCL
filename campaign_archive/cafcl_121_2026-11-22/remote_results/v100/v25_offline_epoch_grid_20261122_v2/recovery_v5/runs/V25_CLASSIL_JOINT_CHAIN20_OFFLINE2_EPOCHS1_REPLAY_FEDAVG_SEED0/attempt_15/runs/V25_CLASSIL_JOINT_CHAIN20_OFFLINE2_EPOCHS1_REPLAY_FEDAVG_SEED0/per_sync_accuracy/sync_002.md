# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.00% on 11 observed labels.
Validation accuracy: 0.1709090909090909; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.21.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **65.5%** (0: 60%, 1: 71%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **37.5%** (1: 71%, 2: 4%) | **7.0%** (6: 0%, 7: 14%) |
| Client 2 | **2.0%** (2: 4%, 3: 0%) | **7.0%** (7: 14%, 8: 0%) |
| Client 3 | **18.0%** (3: 0%, 4: 36%) | **1.5%** (8: 0%, 9: 3%) |
| Client 4 | **18.0%** (4: 36%, 5: 0%) | **23.0%** (9: 3%, 10: 43%) |
