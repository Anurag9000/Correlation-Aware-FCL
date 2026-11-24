# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.18% on 11 observed labels.
Validation accuracy: 0.1890909090909091; seen labels never predicted: 5.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.22181818181818183.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **65.0%** (0: 67%, 1: 63%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **34.5%** (1: 63%, 2: 6%) | **17.0%** (6: 0%, 7: 34%) |
| Client 2 | **3.0%** (2: 6%, 3: 0%) | **17.0%** (7: 34%, 8: 0%) |
| Client 3 | **26.5%** (3: 0%, 4: 53%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **26.5%** (4: 53%, 5: 0%) | **10.5%** (9: 0%, 10: 21%) |
