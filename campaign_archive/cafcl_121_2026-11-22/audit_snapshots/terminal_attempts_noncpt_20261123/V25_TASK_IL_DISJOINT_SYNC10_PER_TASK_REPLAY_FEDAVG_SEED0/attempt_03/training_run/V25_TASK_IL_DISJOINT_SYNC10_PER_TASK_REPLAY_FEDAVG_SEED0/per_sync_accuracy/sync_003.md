# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.00% on 10 observed labels.
Validation accuracy: 0.61; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.6.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **57.5%** (0: 15%, 1: 100%) | **57.5%** (0: 15%, 1: 100%) | **57.5%** (0: 15%, 1: 100%) |
| Client 1 | **57.0%** (2: 25%, 3: 89%) | **57.0%** (2: 25%, 3: 89%) | **57.0%** (2: 25%, 3: 89%) |
| Client 2 | **72.0%** (4: 91%, 5: 53%) | **72.0%** (4: 91%, 5: 53%) | **72.0%** (4: 91%, 5: 53%) |
| Client 3 | **58.0%** (6: 91%, 7: 25%) | **58.0%** (6: 91%, 7: 25%) | **58.0%** (6: 91%, 7: 25%) |
| Client 4 | **55.5%** (8: 50%, 9: 61%) | **55.5%** (8: 50%, 9: 61%) | **55.5%** (8: 50%, 9: 61%) |
