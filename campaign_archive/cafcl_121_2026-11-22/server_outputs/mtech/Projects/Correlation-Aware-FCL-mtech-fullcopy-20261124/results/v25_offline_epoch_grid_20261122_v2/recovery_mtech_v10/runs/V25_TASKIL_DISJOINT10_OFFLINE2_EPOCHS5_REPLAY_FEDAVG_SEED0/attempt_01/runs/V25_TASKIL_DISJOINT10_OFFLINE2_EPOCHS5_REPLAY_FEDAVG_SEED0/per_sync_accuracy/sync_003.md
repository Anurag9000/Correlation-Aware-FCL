# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 63.07% on 30 observed labels.
Validation accuracy: 0.616; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.69.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **73.0%** (0: 52%, 1: 94%) | **55.0%** (10: 97%, 11: 13%) [PENDING upload] | **62.0%** (20: 74%, 21: 50%) [PENDING upload] |
| Client 1 | **78.0%** (2: 56%, 3: 100%) | **58.0%** (12: 93%, 13: 23%) | **69.5%** (22: 44%, 23: 95%) |
| Client 2 | **30.0%** (4: 23%, 5: 37%) [PENDING upload] | **52.0%** (14: 87%, 15: 17%) [PENDING upload] | **57.0%** (24: 54%, 25: 60%) [PENDING upload] |
| Client 3 | **54.5%** (6: 9%, 7: 100%) | **72.0%** (16: 45%, 17: 99%) | **79.0%** (26: 64%, 27: 94%) |
| Client 4 | **65.0%** (8: 98%, 9: 32%) | **71.5%** (18: 89%, 19: 54%) | **69.5%** (28: 84%, 29: 55%) |
