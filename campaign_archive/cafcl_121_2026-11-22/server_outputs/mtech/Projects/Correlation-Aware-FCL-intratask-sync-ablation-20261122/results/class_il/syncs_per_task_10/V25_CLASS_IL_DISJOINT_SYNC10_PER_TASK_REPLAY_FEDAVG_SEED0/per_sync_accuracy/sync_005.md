# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.80% on 10 observed labels.
Validation accuracy: 0.364; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.418.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **51.0%** (0: 57%, 1: 45%) | **51.0%** (0: 57%, 1: 45%) | **51.0%** (0: 57%, 1: 45%) | **51.0%** (0: 57%, 1: 45%) | **51.0%** (0: 57%, 1: 45%) |
| Client 1 | **36.0%** (2: 59%, 3: 13%) | **36.0%** (2: 59%, 3: 13%) | **36.0%** (2: 59%, 3: 13%) | **36.0%** (2: 59%, 3: 13%) | **36.0%** (2: 59%, 3: 13%) |
| Client 2 | **47.5%** (4: 69%, 5: 26%) | **47.5%** (4: 69%, 5: 26%) | **47.5%** (4: 69%, 5: 26%) | **47.5%** (4: 69%, 5: 26%) | **47.5%** (4: 69%, 5: 26%) |
| Client 3 | **31.0%** (6: 41%, 7: 21%) | **31.0%** (6: 41%, 7: 21%) | **31.0%** (6: 41%, 7: 21%) | **31.0%** (6: 41%, 7: 21%) | **31.0%** (6: 41%, 7: 21%) |
| Client 4 | **43.5%** (8: 52%, 9: 35%) | **43.5%** (8: 52%, 9: 35%) | **43.5%** (8: 52%, 9: 35%) | **43.5%** (8: 52%, 9: 35%) | **43.5%** (8: 52%, 9: 35%) |
