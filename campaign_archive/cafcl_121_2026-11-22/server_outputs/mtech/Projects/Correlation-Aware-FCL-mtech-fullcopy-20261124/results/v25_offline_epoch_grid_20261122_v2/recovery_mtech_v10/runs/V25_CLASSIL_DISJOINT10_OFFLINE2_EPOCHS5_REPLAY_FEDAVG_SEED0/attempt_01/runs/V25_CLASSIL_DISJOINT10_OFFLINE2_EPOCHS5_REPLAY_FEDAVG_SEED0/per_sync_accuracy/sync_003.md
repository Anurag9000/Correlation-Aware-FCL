# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 17.13% on 30 observed labels.
Validation accuracy: 0.16466666666666666; seen labels never predicted: 11.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.257.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **40.0%** (0: 40%, 1: 40%) | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **46.5%** (2: 24%, 3: 69%) | **41.0%** (12: 63%, 13: 19%) | **4.0%** (22: 0%, 23: 8%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
| Client 3 | **15.0%** (6: 15%, 7: 15%) | **36.0%** (16: 22%, 17: 50%) | **8.5%** (26: 15%, 27: 2%) |
| Client 4 | **37.0%** (8: 54%, 9: 20%) | **29.0%** (18: 58%, 19: 0%) | **0.0%** (28: 0%, 29: 0%) |
