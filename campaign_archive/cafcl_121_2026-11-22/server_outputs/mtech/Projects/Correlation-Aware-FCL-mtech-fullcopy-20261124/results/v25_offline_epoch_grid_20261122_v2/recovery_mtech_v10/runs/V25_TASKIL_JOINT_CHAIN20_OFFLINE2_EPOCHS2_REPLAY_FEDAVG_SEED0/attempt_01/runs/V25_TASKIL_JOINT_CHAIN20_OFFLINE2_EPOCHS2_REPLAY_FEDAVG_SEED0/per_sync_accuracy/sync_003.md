# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.12% on 16 observed labels.
Validation accuracy: 0.5625; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.625.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **62.0%** (0: 99%, 1: 25%) | **79.0%** (5: 90%, 6: 68%) | **82.0%** (10: 100%, 11: 64%) |
| Client 1 | **60.5%** (1: 25%, 2: 96%) | **54.5%** (6: 68%, 7: 41%) | **46.5%** (11: 64%, 12: 29%) |
| Client 2 | **98.0%** (2: 96%, 3: 100%) | **43.0%** (7: 41%, 8: 45%) | **18.0%** (12: 29%, 13: 7%) |
| Client 3 | **52.0%** (3: 100%, 4: 4%) | **29.0%** (8: 45%, 9: 13%) | **53.0%** (13: 7%, 14: 99%) |
| Client 4 | **47.0%** (4: 4%, 5: 90%) | **56.5%** (9: 13%, 10: 100%) | **50.5%** (14: 99%, 15: 2%) |
