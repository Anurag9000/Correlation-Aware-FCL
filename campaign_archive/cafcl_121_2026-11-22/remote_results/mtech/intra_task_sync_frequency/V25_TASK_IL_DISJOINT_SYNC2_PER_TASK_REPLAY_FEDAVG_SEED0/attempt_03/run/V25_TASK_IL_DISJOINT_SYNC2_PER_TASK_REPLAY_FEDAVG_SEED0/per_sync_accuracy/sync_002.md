# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.40% on 10 observed labels.
Validation accuracy: 0.542; seen labels never predicted: 1.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.544.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **54.5%** (0: 9%, 1: 100%) | **54.5%** (0: 9%, 1: 100%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **50.0%** (2: 0%, 3: 100%) |
| Client 2 | **61.0%** (4: 97%, 5: 25%) | **61.0%** (4: 97%, 5: 25%) |
| Client 3 | **56.0%** (6: 14%, 7: 98%) | **56.0%** (6: 14%, 7: 98%) |
| Client 4 | **50.5%** (8: 99%, 9: 2%) | **50.5%** (8: 99%, 9: 2%) |
