# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.45% on 20 observed labels.
Validation accuracy: 0.543; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5445.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **61.0%** (0: 22%, 1: 100%) | **50.5%** (10: 84%, 11: 17%) |
| Client 1 | **56.5%** (2: 13%, 3: 100%) | **51.5%** (12: 99%, 13: 4%) |
| Client 2 | **56.0%** (4: 100%, 5: 12%) | **50.0%** (14: 0%, 15: 100%) |
| Client 3 | **52.0%** (6: 4%, 7: 100%) | **66.0%** (16: 34%, 17: 98%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **49.5%** (18: 98%, 19: 1%) |
