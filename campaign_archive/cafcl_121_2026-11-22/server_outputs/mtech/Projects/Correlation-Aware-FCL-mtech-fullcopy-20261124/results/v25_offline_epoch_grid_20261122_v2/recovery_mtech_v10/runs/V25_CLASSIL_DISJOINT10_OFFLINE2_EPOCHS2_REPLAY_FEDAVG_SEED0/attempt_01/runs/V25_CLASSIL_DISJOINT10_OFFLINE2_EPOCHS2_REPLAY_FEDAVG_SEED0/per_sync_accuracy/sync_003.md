# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.50% on 30 observed labels.
Validation accuracy: 0.10866666666666666; seen labels never predicted: 12.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.1575.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **20.5%** (0: 22%, 1: 19%) | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **26.5%** (2: 15%, 3: 38%) | **6.0%** (12: 12%, 13: 0%) | **11.5%** (22: 2%, 23: 21%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
| Client 3 | **5.0%** (6: 10%, 7: 0%) | **40.0%** (16: 7%, 17: 73%) | **25.5%** (26: 45%, 27: 6%) |
| Client 4 | **22.0%** (8: 28%, 9: 16%) | **0.0%** (18: 0%, 19: 0%) | **0.5%** (28: 0%, 29: 1%) |
