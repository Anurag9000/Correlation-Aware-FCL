# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.40% on 10 observed labels.
Validation accuracy: 0.76; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.774.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 89%, 1: 88%) | **88.5%** (0: 89%, 1: 88%) | **88.5%** (0: 89%, 1: 88%) |
| Client 1 | **86.0%** (2: 77%, 3: 95%) | **86.0%** (2: 77%, 3: 95%) | **86.0%** (2: 77%, 3: 95%) |
| Client 2 | **80.0%** (4: 85%, 5: 75%) | **80.0%** (4: 85%, 5: 75%) | **80.0%** (4: 85%, 5: 75%) |
| Client 3 | **68.5%** (6: 65%, 7: 72%) | **68.5%** (6: 65%, 7: 72%) | **68.5%** (6: 65%, 7: 72%) |
| Client 4 | **64.0%** (8: 98%, 9: 30%) | **64.0%** (8: 98%, 9: 30%) | **64.0%** (8: 98%, 9: 30%) |
