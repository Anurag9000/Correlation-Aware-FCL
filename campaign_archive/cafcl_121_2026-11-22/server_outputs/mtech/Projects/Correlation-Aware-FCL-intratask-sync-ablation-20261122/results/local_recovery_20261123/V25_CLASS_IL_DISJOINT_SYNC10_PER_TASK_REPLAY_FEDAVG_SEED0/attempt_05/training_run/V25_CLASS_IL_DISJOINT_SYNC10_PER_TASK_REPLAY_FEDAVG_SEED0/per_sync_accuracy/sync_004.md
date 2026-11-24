# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.10% on 10 observed labels.
Validation accuracy: 0.362; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.411.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **57.0%** (0: 57%, 1: 57%) | **57.0%** (0: 57%, 1: 57%) | **57.0%** (0: 57%, 1: 57%) | **57.0%** (0: 57%, 1: 57%) |
| Client 1 | **42.0%** (2: 43%, 3: 41%) | **42.0%** (2: 43%, 3: 41%) | **42.0%** (2: 43%, 3: 41%) | **42.0%** (2: 43%, 3: 41%) |
| Client 2 | **47.5%** (4: 59%, 5: 36%) | **47.5%** (4: 59%, 5: 36%) | **47.5%** (4: 59%, 5: 36%) | **47.5%** (4: 59%, 5: 36%) |
| Client 3 | **23.5%** (6: 38%, 7: 9%) | **23.5%** (6: 38%, 7: 9%) | **23.5%** (6: 38%, 7: 9%) | **23.5%** (6: 38%, 7: 9%) |
| Client 4 | **35.5%** (8: 54%, 9: 17%) | **35.5%** (8: 54%, 9: 17%) | **35.5%** (8: 54%, 9: 17%) | **35.5%** (8: 54%, 9: 17%) |
