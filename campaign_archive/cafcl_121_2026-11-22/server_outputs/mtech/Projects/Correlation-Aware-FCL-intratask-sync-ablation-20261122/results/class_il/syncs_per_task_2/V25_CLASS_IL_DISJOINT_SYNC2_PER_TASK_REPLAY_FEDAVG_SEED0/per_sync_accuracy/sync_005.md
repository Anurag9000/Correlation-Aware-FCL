# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.73% on 30 observed labels.
Validation accuracy: 0.2926666666666667; seen labels never predicted: 9.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.28733333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **64.0%** (0: 69%, 1: 59%) | **64.0%** (0: 69%, 1: 59%) | **27.0%** (10: 19%, 11: 35%) | **27.0%** (10: 19%, 11: 35%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **30.0%** (2: 13%, 3: 47%) | **30.0%** (2: 13%, 3: 47%) | **45.5%** (12: 52%, 13: 39%) | **45.5%** (12: 52%, 13: 39%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **45.0%** (4: 34%, 5: 56%) | **45.0%** (4: 34%, 5: 56%) | **24.5%** (14: 15%, 15: 34%) | **24.5%** (14: 15%, 15: 34%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **48.0%** (6: 47%, 7: 49%) | **48.0%** (6: 47%, 7: 49%) | **57.0%** (16: 32%, 17: 82%) | **57.0%** (16: 32%, 17: 82%) | **0.0%** (26: 0%, 27: 0%) |
| Client 4 | **48.0%** (8: 36%, 9: 60%) | **48.0%** (8: 36%, 9: 60%) | **42.0%** (18: 43%, 19: 41%) | **42.0%** (18: 43%, 19: 41%) | **0.0%** (28: 0%, 29: 0%) |
