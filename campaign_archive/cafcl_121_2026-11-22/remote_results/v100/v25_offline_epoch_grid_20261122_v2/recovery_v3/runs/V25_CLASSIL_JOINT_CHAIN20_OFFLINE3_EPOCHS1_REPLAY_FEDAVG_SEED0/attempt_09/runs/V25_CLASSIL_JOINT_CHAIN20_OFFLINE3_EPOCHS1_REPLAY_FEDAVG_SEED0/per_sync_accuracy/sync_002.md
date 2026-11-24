# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.18% on 11 observed labels.
Validation accuracy: 0.19272727272727272; seen labels never predicted: 5.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.24666666666666667.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **35.5%** (0: 0%, 1: 71%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) [PENDING upload] |
| Client 1 | **55.5%** (1: 71%, 2: 40%) | **3.0%** (6: 0%, 7: 6%) [PENDING upload] |
| Client 2 | **20.0%** (2: 40%, 3: 0%) | **3.0%** (7: 6%, 8: 0%) |
| Client 3 | **25.0%** (3: 0%, 4: 50%) | **11.0%** (8: 0%, 9: 22%) |
| Client 4 | **25.0%** (4: 50%, 5: 0%) | **27.5%** (9: 22%, 10: 33%) |
