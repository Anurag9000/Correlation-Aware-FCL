# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.91% on 11 observed labels.
Validation accuracy: 0.1890909090909091; seen labels never predicted: 4.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.25555555555555554.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **35.5%** (0: 0%, 1: 71%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) [PENDING upload] |
| Client 1 | **75.0%** (1: 71%, 2: 79%) | **0.5%** (6: 0%, 7: 1%) [PENDING upload] |
| Client 2 | **39.5%** (2: 79%, 3: 0%) | **0.5%** (7: 1%, 8: 0%) |
| Client 3 | **25.0%** (3: 0%, 4: 50%) | **5.5%** (8: 0%, 9: 11%) |
| Client 4 | **25.0%** (4: 50%, 5: 0%) | **14.5%** (9: 11%, 10: 18%) |
