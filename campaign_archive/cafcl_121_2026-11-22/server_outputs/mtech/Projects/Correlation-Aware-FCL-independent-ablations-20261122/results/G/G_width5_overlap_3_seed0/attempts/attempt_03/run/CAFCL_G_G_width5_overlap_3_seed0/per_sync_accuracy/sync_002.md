# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 40.96% on 26 observed labels.
Validation accuracy: 0.4107692307692308; seen labels never predicted: 4.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.4096153846153846.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **33.6%** (4: 0%, 7: 62%, 15: 10%, 58: 7%, 63: 89%) | **6.6%** (10: 0%, 26: 13%, 55: 0%, 71: 20%, 90: 0%) |
| Client 1 | **44.4%** (4: 0%, 40: 0%, 69: 56%, 80: 95%, 95: 71%) | **27.4%** (10: 0%, 45: 49%, 55: 0%, 61: 88%, 90: 0%) |
| Client 2 | **17.6%** (4: 0%, 15: 10%, 40: 0%, 58: 7%, 95: 71%) | **20.0%** (10: 0%, 19: 0%, 79: 100%, 82: 0%, 90: 0%) |
| Client 3 | **15.2%** (4: 0%, 16: 5%, 40: 0%, 86: 0%, 95: 71%) | **33.0%** (26: 13%, 55: 0%, 71: 20%, 91: 43%, 96: 89%) |
| Client 4 | **34.6%** (4: 0%, 12: 74%, 16: 5%, 18: 94%, 86: 0%) | **20.0%** (10: 0%, 19: 0%, 55: 0%, 72: 100%, 90: 0%) |
