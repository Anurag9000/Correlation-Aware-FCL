# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 68.90% on 10 observed labels.
Validation accuracy: 0.69; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.689.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **77.0%** (0: 58%, 1: 96%) | **77.0%** (0: 58%, 1: 96%) | **77.0%** (0: 58%, 1: 96%) |
| Client 1 | **76.5%** (2: 54%, 3: 99%) | **76.5%** (2: 54%, 3: 99%) | **76.5%** (2: 54%, 3: 99%) |
| Client 2 | **77.5%** (4: 87%, 5: 68%) | **77.5%** (4: 87%, 5: 68%) | **77.5%** (4: 87%, 5: 68%) |
| Client 3 | **55.0%** (6: 11%, 7: 99%) | **55.0%** (6: 11%, 7: 99%) | **55.0%** (6: 11%, 7: 99%) |
| Client 4 | **58.5%** (8: 97%, 9: 20%) | **58.5%** (8: 97%, 9: 20%) | **58.5%** (8: 97%, 9: 20%) |
