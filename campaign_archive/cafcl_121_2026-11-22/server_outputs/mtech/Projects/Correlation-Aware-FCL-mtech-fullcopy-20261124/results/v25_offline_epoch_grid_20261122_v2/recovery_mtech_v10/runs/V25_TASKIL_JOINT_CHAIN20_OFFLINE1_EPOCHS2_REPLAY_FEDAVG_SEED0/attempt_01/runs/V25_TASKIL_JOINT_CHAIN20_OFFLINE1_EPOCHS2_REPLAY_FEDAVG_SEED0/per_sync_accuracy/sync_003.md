# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.37% on 16 observed labels.
Validation accuracy: 0.58125; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.538125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 89%, 1: 78%) | **54.0%** (5: 87%, 6: 21%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **60.5%** (1: 78%, 2: 43%) | **34.5%** (6: 21%, 7: 48%) | **29.0%** (11: 0%, 12: 58%) |
| Client 2 | **58.5%** (2: 43%, 3: 74%) | **68.0%** (7: 48%, 8: 88%) | **60.5%** (12: 58%, 13: 63%) |
| Client 3 | **37.5%** (3: 74%, 4: 1%) | **44.0%** (8: 88%, 9: 0%) | **46.0%** (13: 63%, 14: 29%) |
| Client 4 | **44.0%** (4: 1%, 5: 87%) | **50.0%** (9: 0%, 10: 100%) | **60.0%** (14: 29%, 15: 91%) |
