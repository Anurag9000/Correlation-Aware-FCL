# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 45.50% on 10 observed labels.
Validation accuracy: 0.408; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.455.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **57.0%** (0: 59%, 1: 55%) | **57.0%** (0: 59%, 1: 55%) | **57.0%** (0: 59%, 1: 55%) | **57.0%** (0: 59%, 1: 55%) |
| Client 1 | **52.0%** (2: 52%, 3: 52%) | **52.0%** (2: 52%, 3: 52%) | **52.0%** (2: 52%, 3: 52%) | **52.0%** (2: 52%, 3: 52%) |
| Client 2 | **38.5%** (4: 32%, 5: 45%) | **38.5%** (4: 32%, 5: 45%) | **38.5%** (4: 32%, 5: 45%) | **38.5%** (4: 32%, 5: 45%) |
| Client 3 | **43.0%** (6: 42%, 7: 44%) | **43.0%** (6: 42%, 7: 44%) | **43.0%** (6: 42%, 7: 44%) | **43.0%** (6: 42%, 7: 44%) |
| Client 4 | **37.0%** (8: 66%, 9: 8%) | **37.0%** (8: 66%, 9: 8%) | **37.0%** (8: 66%, 9: 8%) | **37.0%** (8: 66%, 9: 8%) |
