# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.65% on 20 observed labels.
Validation accuracy: 0.523; seen labels never predicted: 2.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.5265.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **62.0%** (0: 24%, 1: 100%) | **53.0%** (10: 96%, 11: 10%) |
| Client 1 | **53.0%** (2: 6%, 3: 100%) | **50.0%** (12: 100%, 13: 0%) |
| Client 2 | **51.5%** (4: 100%, 5: 3%) | **52.0%** (14: 8%, 15: 96%) |
| Client 3 | **50.5%** (6: 3%, 7: 98%) | **53.0%** (16: 6%, 17: 100%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **50.0%** (18: 100%, 19: 0%) |
