# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.30% on 30 observed labels.
Validation accuracy: 0.6006666666666667; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.647.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **66.5%** (0: 33%, 1: 100%) | **49.5%** (10: 1%, 11: 98%) [PENDING upload] | **72.0%** (20: 94%, 21: 50%) [PENDING upload] |
| Client 1 | **81.5%** (2: 74%, 3: 89%) | **55.5%** (12: 92%, 13: 19%) | **69.5%** (22: 46%, 23: 93%) |
| Client 2 | **40.5%** (4: 17%, 5: 64%) [PENDING upload] | **49.5%** (14: 99%, 15: 0%) [PENDING upload] | **61.0%** (24: 85%, 25: 37%) [PENDING upload] |
| Client 3 | **52.5%** (6: 8%, 7: 97%) | **68.5%** (16: 38%, 17: 99%) | **64.0%** (26: 29%, 27: 99%) |
| Client 4 | **62.0%** (8: 98%, 9: 26%) | **73.0%** (18: 54%, 19: 92%) | **54.0%** (28: 97%, 29: 11%) |
