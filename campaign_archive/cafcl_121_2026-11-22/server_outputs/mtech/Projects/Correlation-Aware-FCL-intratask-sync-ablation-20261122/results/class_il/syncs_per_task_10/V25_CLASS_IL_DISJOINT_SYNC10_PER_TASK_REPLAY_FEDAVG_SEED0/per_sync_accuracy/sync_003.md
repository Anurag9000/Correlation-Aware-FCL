# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.90% on 10 observed labels.
Validation accuracy: 0.268; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.259.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **29.5%** (0: 51%, 1: 8%) | **29.5%** (0: 51%, 1: 8%) | **29.5%** (0: 51%, 1: 8%) |
| Client 1 | **22.5%** (2: 5%, 3: 40%) | **22.5%** (2: 5%, 3: 40%) | **22.5%** (2: 5%, 3: 40%) |
| Client 2 | **24.5%** (4: 44%, 5: 5%) | **24.5%** (4: 44%, 5: 5%) | **24.5%** (4: 44%, 5: 5%) |
| Client 3 | **7.5%** (6: 14%, 7: 1%) | **7.5%** (6: 14%, 7: 1%) | **7.5%** (6: 14%, 7: 1%) |
| Client 4 | **45.5%** (8: 59%, 9: 32%) | **45.5%** (8: 59%, 9: 32%) | **45.5%** (8: 59%, 9: 32%) |
