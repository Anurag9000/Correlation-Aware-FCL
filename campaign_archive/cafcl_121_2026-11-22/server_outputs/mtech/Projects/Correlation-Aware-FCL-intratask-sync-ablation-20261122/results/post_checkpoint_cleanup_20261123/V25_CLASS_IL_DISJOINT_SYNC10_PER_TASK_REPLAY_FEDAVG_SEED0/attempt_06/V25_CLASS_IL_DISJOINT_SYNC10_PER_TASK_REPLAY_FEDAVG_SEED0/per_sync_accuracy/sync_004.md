# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 36.90% on 10 observed labels.
Validation accuracy: 0.32; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.369.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **59.5%** (0: 53%, 1: 66%) | **59.5%** (0: 53%, 1: 66%) | **59.5%** (0: 53%, 1: 66%) | **59.5%** (0: 53%, 1: 66%) |
| Client 1 | **28.0%** (2: 23%, 3: 33%) | **28.0%** (2: 23%, 3: 33%) | **28.0%** (2: 23%, 3: 33%) | **28.0%** (2: 23%, 3: 33%) |
| Client 2 | **40.5%** (4: 56%, 5: 25%) | **40.5%** (4: 56%, 5: 25%) | **40.5%** (4: 56%, 5: 25%) | **40.5%** (4: 56%, 5: 25%) |
| Client 3 | **21.5%** (6: 35%, 7: 8%) | **21.5%** (6: 35%, 7: 8%) | **21.5%** (6: 35%, 7: 8%) | **21.5%** (6: 35%, 7: 8%) |
| Client 4 | **35.0%** (8: 61%, 9: 9%) | **35.0%** (8: 61%, 9: 9%) | **35.0%** (8: 61%, 9: 9%) | **35.0%** (8: 61%, 9: 9%) |
