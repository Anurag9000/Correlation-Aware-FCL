# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.50% on 10 observed labels.
Validation accuracy: 0.374; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.415.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **57.0%** (0: 56%, 1: 58%) | **57.0%** (0: 56%, 1: 58%) | **57.0%** (0: 56%, 1: 58%) | **57.0%** (0: 56%, 1: 58%) | **57.0%** (0: 56%, 1: 58%) |
| Client 1 | **29.0%** (2: 52%, 3: 6%) | **29.0%** (2: 52%, 3: 6%) | **29.0%** (2: 52%, 3: 6%) | **29.0%** (2: 52%, 3: 6%) | **29.0%** (2: 52%, 3: 6%) |
| Client 2 | **51.0%** (4: 68%, 5: 34%) | **51.0%** (4: 68%, 5: 34%) | **51.0%** (4: 68%, 5: 34%) | **51.0%** (4: 68%, 5: 34%) | **51.0%** (4: 68%, 5: 34%) |
| Client 3 | **30.5%** (6: 39%, 7: 22%) | **30.5%** (6: 39%, 7: 22%) | **30.5%** (6: 39%, 7: 22%) | **30.5%** (6: 39%, 7: 22%) | **30.5%** (6: 39%, 7: 22%) |
| Client 4 | **40.0%** (8: 32%, 9: 48%) | **40.0%** (8: 32%, 9: 48%) | **40.0%** (8: 32%, 9: 48%) | **40.0%** (8: 32%, 9: 48%) | **40.0%** (8: 32%, 9: 48%) |
