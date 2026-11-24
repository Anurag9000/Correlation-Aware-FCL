# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 46.70% on 10 observed labels.
Validation accuracy: 0.412; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.467.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **68.0%** (0: 77%, 1: 59%) | **68.0%** (0: 77%, 1: 59%) | **68.0%** (0: 77%, 1: 59%) | **68.0%** (0: 77%, 1: 59%) | **68.0%** (0: 77%, 1: 59%) | **68.0%** (0: 77%, 1: 59%) |
| Client 1 | **44.0%** (2: 48%, 3: 40%) | **44.0%** (2: 48%, 3: 40%) | **44.0%** (2: 48%, 3: 40%) | **44.0%** (2: 48%, 3: 40%) | **44.0%** (2: 48%, 3: 40%) | **44.0%** (2: 48%, 3: 40%) |
| Client 2 | **44.0%** (4: 62%, 5: 26%) | **44.0%** (4: 62%, 5: 26%) | **44.0%** (4: 62%, 5: 26%) | **44.0%** (4: 62%, 5: 26%) | **44.0%** (4: 62%, 5: 26%) | **44.0%** (4: 62%, 5: 26%) |
| Client 3 | **34.5%** (6: 43%, 7: 26%) | **34.5%** (6: 43%, 7: 26%) | **34.5%** (6: 43%, 7: 26%) | **34.5%** (6: 43%, 7: 26%) | **34.5%** (6: 43%, 7: 26%) | **34.5%** (6: 43%, 7: 26%) |
| Client 4 | **43.0%** (8: 62%, 9: 24%) | **43.0%** (8: 62%, 9: 24%) | **43.0%** (8: 62%, 9: 24%) | **43.0%** (8: 62%, 9: 24%) | **43.0%** (8: 62%, 9: 24%) | **43.0%** (8: 62%, 9: 24%) |
