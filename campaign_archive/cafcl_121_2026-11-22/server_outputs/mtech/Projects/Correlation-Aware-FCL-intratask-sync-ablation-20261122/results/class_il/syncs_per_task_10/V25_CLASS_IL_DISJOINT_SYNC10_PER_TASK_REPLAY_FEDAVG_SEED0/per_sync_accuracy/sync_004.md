# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.00% on 10 observed labels.
Validation accuracy: 0.308; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.34.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 54%, 1: 55%) | **54.5%** (0: 54%, 1: 55%) | **54.5%** (0: 54%, 1: 55%) | **54.5%** (0: 54%, 1: 55%) |
| Client 1 | **16.0%** (2: 24%, 3: 8%) | **16.0%** (2: 24%, 3: 8%) | **16.0%** (2: 24%, 3: 8%) | **16.0%** (2: 24%, 3: 8%) |
| Client 2 | **36.0%** (4: 61%, 5: 11%) | **36.0%** (4: 61%, 5: 11%) | **36.0%** (4: 61%, 5: 11%) | **36.0%** (4: 61%, 5: 11%) |
| Client 3 | **20.5%** (6: 37%, 7: 4%) | **20.5%** (6: 37%, 7: 4%) | **20.5%** (6: 37%, 7: 4%) | **20.5%** (6: 37%, 7: 4%) |
| Client 4 | **43.0%** (8: 70%, 9: 16%) | **43.0%** (8: 70%, 9: 16%) | **43.0%** (8: 70%, 9: 16%) | **43.0%** (8: 70%, 9: 16%) |
