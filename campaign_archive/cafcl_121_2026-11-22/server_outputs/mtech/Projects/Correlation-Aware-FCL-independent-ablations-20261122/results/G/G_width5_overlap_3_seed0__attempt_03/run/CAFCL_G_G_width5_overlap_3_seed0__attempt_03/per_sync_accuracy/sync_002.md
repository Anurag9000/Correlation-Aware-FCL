# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 40.00% on 26 observed labels.
Validation accuracy: 0.3953846153846154; seen labels never predicted: 4.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.4.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **32.8%** (4: 0%, 7: 52%, 15: 14%, 58: 8%, 63: 90%) | **12.4%** (10: 0%, 26: 26%, 55: 0%, 71: 36%, 90: 0%) |
| Client 1 | **44.0%** (4: 0%, 40: 0%, 69: 60%, 80: 96%, 95: 64%) | **28.8%** (10: 0%, 45: 58%, 55: 0%, 61: 86%, 90: 0%) |
| Client 2 | **17.2%** (4: 0%, 15: 14%, 40: 0%, 58: 8%, 95: 64%) | **20.0%** (10: 0%, 19: 0%, 79: 100%, 82: 0%, 90: 0%) |
| Client 3 | **16.4%** (4: 0%, 16: 17%, 40: 0%, 86: 1%, 95: 64%) | **27.0%** (26: 26%, 55: 0%, 71: 36%, 91: 60%, 96: 13%) |
| Client 4 | **35.4%** (4: 0%, 12: 60%, 16: 17%, 18: 99%, 86: 1%) | **20.0%** (10: 0%, 19: 0%, 55: 0%, 72: 100%, 90: 0%) |
