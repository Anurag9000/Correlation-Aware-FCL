# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.20% on 10 observed labels.
Validation accuracy: 0.158; seen labels never predicted: 3.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.162.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **1.0%** (0: 2%, 1: 0%) | **1.0%** (0: 2%, 1: 0%) |
| Client 1 | **0.5%** (2: 1%, 3: 0%) | **0.5%** (2: 1%, 3: 0%) |
| Client 2 | **39.0%** (4: 76%, 5: 2%) | **39.0%** (4: 76%, 5: 2%) |
| Client 3 | **9.5%** (6: 0%, 7: 19%) | **9.5%** (6: 0%, 7: 19%) |
| Client 4 | **31.0%** (8: 62%, 9: 0%) | **31.0%** (8: 62%, 9: 0%) |
