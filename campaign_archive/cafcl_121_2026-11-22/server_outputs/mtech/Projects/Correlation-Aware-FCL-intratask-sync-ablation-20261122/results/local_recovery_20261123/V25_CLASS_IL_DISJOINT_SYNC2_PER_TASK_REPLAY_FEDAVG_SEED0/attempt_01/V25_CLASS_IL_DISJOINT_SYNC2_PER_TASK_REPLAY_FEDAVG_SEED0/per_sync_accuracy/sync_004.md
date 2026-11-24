# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 36.30% on 20 observed labels.
Validation accuracy: 0.35; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.363.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **51.0%** (0: 61%, 1: 41%) | **51.0%** (0: 61%, 1: 41%) | **12.0%** (10: 12%, 11: 12%) | **12.0%** (10: 12%, 11: 12%) |
| Client 1 | **30.5%** (2: 12%, 3: 49%) | **30.5%** (2: 12%, 3: 49%) | **50.0%** (12: 74%, 13: 26%) | **50.0%** (12: 74%, 13: 26%) |
| Client 2 | **47.0%** (4: 38%, 5: 56%) | **47.0%** (4: 38%, 5: 56%) | **17.5%** (14: 8%, 15: 27%) | **17.5%** (14: 8%, 15: 27%) |
| Client 3 | **40.0%** (6: 31%, 7: 49%) | **40.0%** (6: 31%, 7: 49%) | **45.0%** (16: 28%, 17: 62%) | **45.0%** (16: 28%, 17: 62%) |
| Client 4 | **33.0%** (8: 29%, 9: 37%) | **33.0%** (8: 29%, 9: 37%) | **37.0%** (18: 52%, 19: 22%) | **37.0%** (18: 52%, 19: 22%) |
