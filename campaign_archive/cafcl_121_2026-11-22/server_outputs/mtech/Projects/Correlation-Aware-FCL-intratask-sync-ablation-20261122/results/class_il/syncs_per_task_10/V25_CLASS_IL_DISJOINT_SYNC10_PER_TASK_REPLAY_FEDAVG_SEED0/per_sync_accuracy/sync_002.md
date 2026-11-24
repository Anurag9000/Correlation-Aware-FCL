# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 11.10% on 10 observed labels.
Validation accuracy: 0.118; seen labels never predicted: 5.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.111.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **3.5%** (0: 7%, 1: 0%) | **3.5%** (0: 7%, 1: 0%) |
| Client 1 | **1.5%** (2: 0%, 3: 3%) | **1.5%** (2: 0%, 3: 3%) |
| Client 2 | **50.0%** (4: 96%, 5: 4%) | **50.0%** (4: 96%, 5: 4%) |
| Client 3 | **0.5%** (6: 1%, 7: 0%) | **0.5%** (6: 1%, 7: 0%) |
| Client 4 | **0.0%** (8: 0%, 9: 0%) | **0.0%** (8: 0%, 9: 0%) |
