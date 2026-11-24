# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.27% on 11 observed labels.
Validation accuracy: 0.1890909090909091; seen labels never predicted: 5.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.23555555555555555.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **32.0%** (0: 0%, 1: 64%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) [PENDING upload] |
| Client 1 | **63.5%** (1: 64%, 2: 63%) | **1.0%** (6: 0%, 7: 2%) [PENDING upload] |
| Client 2 | **31.5%** (2: 63%, 3: 0%) | **1.0%** (7: 2%, 8: 0%) |
| Client 3 | **19.5%** (3: 0%, 4: 39%) | **16.0%** (8: 0%, 9: 32%) |
| Client 4 | **19.5%** (4: 39%, 5: 0%) | **22.0%** (9: 32%, 10: 12%) |
