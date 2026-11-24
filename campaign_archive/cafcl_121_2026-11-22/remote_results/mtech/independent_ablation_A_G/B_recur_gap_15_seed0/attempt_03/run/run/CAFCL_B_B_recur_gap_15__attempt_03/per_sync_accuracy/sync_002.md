# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.40% on 20 observed labels.
Validation accuracy: 0.603; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.594.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **68.5%** (0: 37%, 1: 100%) | **52.0%** (10: 99%, 11: 5%) |
| Client 1 | **60.0%** (2: 20%, 3: 100%) | **60.0%** (12: 23%, 13: 97%) |
| Client 2 | **58.5%** (4: 98%, 5: 19%) | **54.5%** (14: 9%, 15: 100%) |
| Client 3 | **64.0%** (6: 63%, 7: 65%) | **65.5%** (16: 33%, 17: 98%) |
| Client 4 | **52.5%** (8: 99%, 9: 6%) | **58.5%** (18: 94%, 19: 23%) |
