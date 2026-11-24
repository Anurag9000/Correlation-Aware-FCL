# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.73% on 30 observed labels.
Validation accuracy: 0.206; seen labels never predicted: 13.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.311.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **47.0%** (0: 44%, 1: 50%) | **0.0%** (10: 0%, 11: 0%) [PENDING upload] | **0.0%** (20: 0%, 21: 0%) [PENDING upload] |
| Client 1 | **48.0%** (2: 31%, 3: 65%) | **36.0%** (12: 34%, 13: 38%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] | **0.0%** (14: 0%, 15: 0%) [PENDING upload] | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
| Client 3 | **45.5%** (6: 41%, 7: 50%) | **31.5%** (16: 6%, 17: 57%) | **1.5%** (26: 2%, 27: 1%) |
| Client 4 | **69.0%** (8: 86%, 9: 52%) | **32.5%** (18: 48%, 19: 17%) | **0.0%** (28: 0%, 29: 0%) |
