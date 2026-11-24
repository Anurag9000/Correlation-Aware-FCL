# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 45.60% on 10 observed labels.
Validation accuracy: 0.404; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.456.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **65.0%** (0: 76%, 1: 54%) | **65.0%** (0: 76%, 1: 54%) | **65.0%** (0: 76%, 1: 54%) | **65.0%** (0: 76%, 1: 54%) | **65.0%** (0: 76%, 1: 54%) | **65.0%** (0: 76%, 1: 54%) |
| Client 1 | **36.5%** (2: 46%, 3: 27%) | **36.5%** (2: 46%, 3: 27%) | **36.5%** (2: 46%, 3: 27%) | **36.5%** (2: 46%, 3: 27%) | **36.5%** (2: 46%, 3: 27%) | **36.5%** (2: 46%, 3: 27%) |
| Client 2 | **45.0%** (4: 70%, 5: 20%) | **45.0%** (4: 70%, 5: 20%) | **45.0%** (4: 70%, 5: 20%) | **45.0%** (4: 70%, 5: 20%) | **45.0%** (4: 70%, 5: 20%) | **45.0%** (4: 70%, 5: 20%) |
| Client 3 | **36.5%** (6: 51%, 7: 22%) | **36.5%** (6: 51%, 7: 22%) | **36.5%** (6: 51%, 7: 22%) | **36.5%** (6: 51%, 7: 22%) | **36.5%** (6: 51%, 7: 22%) | **36.5%** (6: 51%, 7: 22%) |
| Client 4 | **45.0%** (8: 68%, 9: 22%) | **45.0%** (8: 68%, 9: 22%) | **45.0%** (8: 68%, 9: 22%) | **45.0%** (8: 68%, 9: 22%) | **45.0%** (8: 68%, 9: 22%) | **45.0%** (8: 68%, 9: 22%) |
