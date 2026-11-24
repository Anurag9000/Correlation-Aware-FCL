# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.70% on 10 observed labels.
Validation accuracy: 0.246; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.247.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **28.0%** (0: 39%, 1: 17%) | **28.0%** (0: 39%, 1: 17%) | **28.0%** (0: 39%, 1: 17%) |
| Client 1 | **17.5%** (2: 18%, 3: 17%) | **17.5%** (2: 18%, 3: 17%) | **17.5%** (2: 18%, 3: 17%) |
| Client 2 | **27.0%** (4: 54%, 5: 0%) | **27.0%** (4: 54%, 5: 0%) | **27.0%** (4: 54%, 5: 0%) |
| Client 3 | **5.0%** (6: 10%, 7: 0%) | **5.0%** (6: 10%, 7: 0%) | **5.0%** (6: 10%, 7: 0%) |
| Client 4 | **46.0%** (8: 75%, 9: 17%) | **46.0%** (8: 75%, 9: 17%) | **46.0%** (8: 75%, 9: 17%) |
